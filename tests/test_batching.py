import pytest
import torch
from yue2.batching import BatchGraphAR, generate_tokens_batch
from yue2.cuda_graph import GraphAR
from yue2.modeling_yue2 import YuE2Config,YuE2ForCausalLM
from yue2.protocol import Sampling,ABC_END,VOCAB_SIZE

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_independent_rows_match_separate_graphs(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    previous=torch.get_num_threads();torch.set_num_threads(1)
    graphs=[]
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(801)
            model=YuE2ForCausalLM(YuE2Config(hidden_size=128,intermediate_size=256,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,head_dim=32,vocab_size=32,
                max_position_embeddings=64,max_latent_frames=64)).eval().to(device=device,dtype=torch.bfloat16 if device=='cuda' else torch.float32)
        prefixes=[[2,3,4,5],[6]]
        batch=BatchGraphAR(model,prefixes,4,capture=device=='cuda');graphs.append(batch)
        singles=[GraphAR(model,[p],4,capture=device=='cuda') for p in prefixes];graphs.extend(singles)
        atol,rtol=(.02,.03) if device=='cuda' else (1e-6,1e-5)
        torch.testing.assert_close(batch.prefill(),torch.cat([g.prefill() for g in singles]),atol=atol,rtol=rtol)
        for tokens in [[7,9],[8,10],[12,11]]:
            got=batch.step_batch(torch.tensor(tokens,device=device).reshape(2,1))
            expected=torch.cat([g.step(t) for g,t in zip(singles,tokens)])
            torch.testing.assert_close(got,expected,atol=atol,rtol=rtol)
        assert batch.positions.tolist()==[7,4]
        with pytest.raises(ValueError,match='budget'):batch.step_batch(torch.tensor([[1],[2]]))
    finally:
        for g in graphs:g.close()
        torch.set_num_threads(previous)


def test_eos_is_independent_and_graph_is_closed(monkeypatch):
    import yue2.batching as module
    instances=[]
    class FakeGraph:
        attention_backend='test'
        def __init__(self,*args,**kwargs):self.steps=0;self.closed=False;instances.append(self)
        def logits(self,ids):
            x=torch.full((2,VOCAB_SIZE),-100.)
            for row,t in enumerate(ids):x[row,t]=100
            return x
        def prefill(self):return self.logits([ABC_END,3])
        def step_batch(self,tokens):self.steps+=1;return self.logits([4,ABC_END])
        def close(self):self.closed=True
    monkeypatch.setattr(module,'BatchGraphAR',FakeGraph)
    seen=[]
    rows=generate_tokens_batch(torch.nn.Linear(1,1),[[1],[2,3]],Sampling(temperature=0,min_tokens=0,max_tokens=4),[7,9],'abc',on_token=lambda *x:seen.append(x))
    assert [r[0] for r in rows]==[[],[3]]
    assert [r[2] for r in rows]==[False,False]
    assert [r[1]['output_tokens'] for r in rows]==[1,2]
    assert seen==[(0,'abc',ABC_END),(1,'abc',3),(1,'abc',ABC_END)]
    assert instances[0].closed and instances[0].steps==1


def test_sampling_rng_is_independent_and_cancellation_closes(monkeypatch):
    import yue2.batching as module
    instances=[]
    class FakeGraph:
        attention_backend='test'
        def __init__(self,*a,**k):self.closed=False;instances.append(self)
        def prefill(self):
            logits=torch.full((2,VOCAB_SIZE),-torch.inf)
            logits[:,:9]=0
            return logits
        def step_batch(self,tokens):return self.prefill()
        def close(self):self.closed=True
    monkeypatch.setattr(module,'BatchGraphAR',FakeGraph)
    cfg=Sampling(temperature=1,top_k=9,top_p=1,repetition_penalty=1,min_tokens=0,max_tokens=5)
    model=torch.nn.Linear(1,1)
    a=generate_tokens_batch(model,[[1],[2]],cfg,[15,28],'abc')
    b=generate_tokens_batch(model,[[2],[1]],cfg,[28,15],'abc')
    assert a[0][0]==b[1][0] and a[1][0]==b[0][0]
    assert all(r[2] and len(r[0])==5 for r in a)
    calls=[]
    with pytest.raises(InterruptedError):
        generate_tokens_batch(model,[[1],[2]],cfg,[15,28],'abc',cancelled=lambda:len(calls)>0,on_token=lambda *x:calls.append(x))
    assert instances[-1].closed


@pytest.mark.parametrize('change',[
    {'backend':'vllm'}, {'resident_models':False}, {'offload_ar':True},
    {'quantization':'int8'}, {'device':torch.device('cpu')},
])
def test_unsupported_pipeline_rejected_before_loading(change):
    from types import SimpleNamespace
    from yue2.batching import generate_batch
    settings=dict(backend='torch',resident_models=True,offload_ar=False,
                  quantization='none',device=torch.device('cuda'))
    settings.update(change)
    pipe=SimpleNamespace(**settings)
    with pytest.raises(ValueError,match='requires CUDA'):
        generate_batch(pipe,[dict(style='pop',lyrics='hello')]*2)


def test_request_count_cfg_and_early_cancellation():
    from types import SimpleNamespace
    from yue2.batching import generate_batch
    pipe=SimpleNamespace(backend='torch',resident_models=True,offload_ar=False,
                         quantization='none',device=torch.device('cuda'))
    request=dict(style='pop',lyrics='hello')
    with pytest.raises(ValueError,match='exactly two'):
        generate_batch(pipe,[request])
    with pytest.raises(ValueError,match='cfg_scale=1'):
        generate_batch(pipe,[dict(request,cfg_scale=2),request])
    with pytest.raises(ValueError,match='cot=full/melody'):
        generate_batch(pipe,[dict(request,cot='off'),request])
    with pytest.raises(InterruptedError,match='before batch'):
        generate_batch(pipe,[request,request],cancelled=lambda:True)
