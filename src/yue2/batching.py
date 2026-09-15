"""Experimental batch=2 AR generation; independent requests, not CFG branches.

NAR synthesis and VAE decoding remain sequential. This is an opt-in pipeline
API, not a thread-safe wrapper or a change to the HTTP worker's admission queue.
"""
from __future__ import annotations
import time
import torch
from .cuda_graph import GraphAR
from .protocol import ABC_END, MUSIC_END, CODEC_OFFSET, SongRequest, token_prefixes, resolve_sampling
from .sampling import distribution, synchronize


class BatchGraphAR(GraphAR):
    """Use GraphAR's branch-isolated KV storage with independent row tokens."""
    def __init__(self, model, prefixes, max_tokens, **kwargs):
        if len(prefixes) != 2:
            raise ValueError('BatchGraphAR requires exactly two independent requests')
        super().__init__(model, prefixes, max_tokens, **kwargs)

    @torch.inference_mode()
    def step_batch(self, tokens):
        if self.closed or not self.ready:
            raise RuntimeError('Call prefill before step_batch on an open graph')
        if self.steps >= self.max_tokens - 1:
            raise ValueError('Requested generation budget is exhausted')
        if not isinstance(tokens, torch.Tensor) or tokens.dtype not in {torch.int32, torch.int64} or tokens.shape != (2, 1):
            raise ValueError('Expected integer tokens of shape [2,1]')
        if tokens.device.type == 'cpu' and (tokens.min().item() < 0 or tokens.max().item() >= self.model.config.vocab_size):
            raise ValueError('Token is outside the model vocabulary')
        self.tokens.copy_(tokens)
        if self.graph is not None:
            self.graph.replay()
            result = self.output
        else:
            result = self._decode()
        self.steps += 1
        return result


@torch.inference_mode()
def generate_tokens_batch(model, prefixes, sampling, seeds, phase, *, cancelled=None, on_token=None):
    """One forward for two active rows; finished rows use ignored dummy tokens.

    No compaction or continuous admission: cache positions for finished rows
    can advance but cannot exceed the shared preallocated generation budget.
    Each live row samples alone with its own RNG, penalty history and EOS.
    Cancellation aborts the whole batch; per-job cancellation is not supported.
    """
    if len(prefixes) != 2 or len(seeds) != 2 or phase not in {'abc', 'semantic'}:
        raise ValueError('Require two prefixes/seeds and phase abc or semantic')
    if cancelled is not None and cancelled():
        raise InterruptedError('Cancelled before batch prefill')
    device = next(model.parameters()).device
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    histories = [[], []]; ended = [False, False]; first = [None, None]; completed = [None, None]
    graph = None; end = ABC_END if phase == 'abc' else MUSIC_END
    synchronize(device); started = time.perf_counter()
    try:
        graph = BatchGraphAR(model, prefixes, sampling.max_tokens, capture=device.type == 'cuda')
        logits = graph.prefill(); synchronize(device)
        prefill_seconds = time.perf_counter() - started
        for step in range(sampling.max_tokens):
            if cancelled is not None and cancelled():
                raise InterruptedError('Cancelled during batched '+phase)
            next_tokens = torch.zeros((2, 1), dtype=torch.long, device=device)
            for row in range(2):
                if ended[row]:
                    continue
                scores = distribution(logits[row:row+1], sampling, histories[row], step, phase)
                token_tensor = scores.argmax(-1, keepdim=True) if sampling.temperature == 0 else torch.multinomial(
                    scores.softmax(-1), 1, generator=generators[row])
                next_tokens[row:row+1].copy_(token_tensor)
                token = int(token_tensor.item())
                elapsed = time.perf_counter() - started
                if first[row] is None: first[row] = elapsed
                if on_token is not None: on_token(row, phase, token)
                if token == end:
                    ended[row] = True; completed[row] = elapsed
                else:
                    histories[row].append(token)
            if all(ended): break
            if step + 1 < sampling.max_tokens:
                logits = graph.step_batch(next_tokens)
        synchronize(device); seconds = time.perf_counter() - started
        return [(histories[row], {'seconds': seconds, 'shared_batch_seconds': seconds,
                    'row_finished_seconds': completed[row], 'prefill_seconds': prefill_seconds,
                    'ttft_seconds': first[row], 'output_tokens': len(histories[row])+int(ended[row]),
                    'content_tokens': len(histories[row]), 'prefix_tokens': len(prefixes[row]),
                    'execution': 'cuda_graph_batch2' if device.type == 'cuda' else 'eager_batch2',
                    'attention': graph.attention_backend, 'batch_size': 2, 'cfg_branches': 1}, not ended[row])
                for row in range(2)]
    finally:
        if graph is not None: graph.close()


def generate_batch(pipe, requests, *, abc_sampling=None, semantic_sampling=None, cancelled=None, on_token=None, on_stage=None):
    """Generate exactly two songs with shared AR forwards and sequential NAR/VAE.

    Supported: torch, CUDA, resident BF16, cot=full/melody, CFG=1. Full results
    return together after both songs finish. Arbitrary concurrent calls on the
    same pipeline are not safe. Batch shape can change floating-point rounding.
    """
    from .pipeline import SymbolicPlan, SemanticResult, SongResult
    from .storage import identity
    requests = [r if isinstance(r, SongRequest) else SongRequest(**r) for r in requests]
    if len(requests) != 2:
        raise ValueError('generate_batch requires exactly two requests')
    if any(r.cot not in {'full','melody'} or r.guidance != 1 for r in requests):
        raise ValueError('Batch prototype requires cot=full/melody and cfg_scale=1')
    if pipe.backend != 'torch' or pipe.quantization != 'none' or pipe.offload_ar or not pipe.resident_models or pipe.device.type != 'cuda':
        raise ValueError('Batch prototype requires CUDA torch, unquantized resident models, and offload_ar=False')
    if cancelled is not None and cancelled(): raise InterruptedError('Cancelled before batch')
    abc_sampling = resolve_sampling(abc_sampling, pipe.generation_config.abc)
    semantic_sampling = resolve_sampling(semantic_sampling, pipe.generation_config.semantic)
    started = time.perf_counter(); model = pipe._load_model(); plans = [None,None]
    def stage(name):
        if on_stage is not None: on_stage(name)
    stage('planning')
    if all(r.abc is None for r in requests):
        rows = generate_tokens_batch(model, [token_prefixes(r,pipe.tokenizer) for r in requests],
                     abc_sampling,[r.seed for r in requests],'abc',cancelled=cancelled,on_token=on_token)
        for index,(ids,timing,truncated) in enumerate(rows):
            r=requests[index];plans[index]=SymbolicPlan(r,pipe.tokenizer.decode(ids),ids,token_prefixes(r,pipe.tokenizer,ids),timing,truncated)
    else:
        # External scores do not need generation. A lone missing score uses
        # the existing single-request planner, retaining exact input tokens.
        for index,r in enumerate(requests):
            callback = (lambda phase,token,i=index:on_token(i,phase,token)) if on_token is not None else None
            plans[index]=pipe.plan(request=r,abc_sampling=abc_sampling,cancelled=cancelled,on_token=callback)
    stage('semantic')
    rows=generate_tokens_batch(model,[p.prefix for p in plans],semantic_sampling,[r.seed for r in requests],
                               'semantic',cancelled=cancelled,on_token=on_token)
    results=[]
    for index,(ids,timing,truncated) in enumerate(rows):
        semantic=SemanticResult(plans[index],[int(t)-CODEC_OFFSET for t in ids],timing,truncated)
        stage(f'synthesis:{index}');t=time.perf_counter();latents=pipe.synthesize(semantic,cancelled=cancelled);nar=time.perf_counter()-t
        stage(f'decode:{index}');t=time.perf_counter();audio=pipe.decode(latents,cancelled=cancelled);vae=time.perf_counter()-t
        config=pipe.effective_config(requests[index],abc_sampling,semantic_sampling)
        config['batch']={'size':2,'ar':'shared_forward','nar':'sequential','vae':'sequential','continuous_batching':False,
                         'cancellation':'whole_batch','validation_status':'experimental'}
        request_id=identity({'request':requests[index].to_dict(),'config':config,'weights':pipe.weights})
        results.append(SongResult(audio,48000,semantic,latents,config,pipe.weights,
                       {'abc':plans[index].timing,'semantic':timing,'nar_seconds':nar,'vae_seconds':vae,
                        'batch_item_ready_seconds':time.perf_counter()-started,'load':dict(pipe.load_timing)},request_id))
    synchronize(pipe.device);total=time.perf_counter()-started
    for result in results: result.timing['batch_return_seconds']=total
    return results
