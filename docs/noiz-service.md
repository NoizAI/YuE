# Noiz YuE2 服务与 5090 推理优化

此分支增加 FastAPI 异步服务和显存常驻优化。部署目标：Linux、标准 RTX 5090 32 GB、
Python 3.12、PyTorch 2.10.0 CUDA 12.8 构建。真实 GPU 测试完成前不承诺耗时或加速比例。
这是单机、单 GPU、单租户服务：持有 API 密钥的调用方可读取本服务中的全部任务。

## 在 5090 上运行

先安装支持 5090 的 NVIDIA 驱动，确认 `nvidia-smi` 正常。
Docker 路径还需要 Docker Compose（支持 `gpus`）与 NVIDIA Container Toolkit。

```bash
git clone https://github.com/NoizAI/YuE.git
cd YuE
git switch codex/fastapi-inference-acceleration
cp .env.example .env
# 编辑 .env：用 openssl rand -hex 32 的结果替换 YUE2_API_KEY。
docker compose up --build -d
docker compose logs -f yue2
curl -i http://127.0.0.1:8000/health/ready
```

镜像不含模型权重，首次启动下载到持久化卷并预热。`/health/ready` 返回 200 后接任务，
加载或失败时返回 503；`/health/live` 只表示 HTTP 进程存活。
镜像构建与 CUDA 执行需要在 Linux/amd64 验证，CPU 单测不能替代它。
默认只映射本机端口。跨机器调用时通过业务网关提供 HTTPS 和访问控制。

也可以直接在 Linux 主机安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install '.[server]'
export YUE2_API_KEY="$(openssl rand -hex 32)"
export YUE2_DATA_DIR="$PWD/outputs/service"
yue2-serve
```

`.env` 由 Compose 读取；直接运行时自行导出环境变量。保留密钥给调用方，不要提交到 Git。
可选配置：`YUE2_MODEL` / `YUE2_VAE`、`YUE2_REVISION` / `YUE2_VAE_REVISION` 指定模型和版本；
`YUE2_LOCAL_FILES_ONLY=true` 禁止自动下载。Compose 使用这些可选项时要补入 environment。

## 接口

Swagger：`http://127.0.0.1:8000/docs`，点击 Authorize 输入密钥。
健康检查和 OpenAPI 文档公开，所有业务接口需要 Bearer token。

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Authorization: Bearer $YUE2_API_KEY" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: my-business-order-001' \
  -d '{"style":"Mandarin, piano pop, warm vocal", "lyrics":"[Verse]\n晨光落在窗边\n我们走向新的一天\n[Chorus]\n让歌声陪伴你", "cot":"full", "seed":42}'
```

返回 202，含 `id`、`status=queued`；Location 指向状态接口。
相同幂等键和相同输入重试返回原任务（200），不同输入返回 409；幂等键不自动过期。
再次生成需用新键，接口每次只生成一个候选。

```bash
JOB_ID=替换为返回的id
curl -H "Authorization: Bearer $YUE2_API_KEY" "http://127.0.0.1:8000/v1/jobs/$JOB_ID"
curl -H "Authorization: Bearer $YUE2_API_KEY" "http://127.0.0.1:8000/v1/jobs/$JOB_ID/audio" -o song.flac
curl -H "Authorization: Bearer $YUE2_API_KEY" "http://127.0.0.1:8000/v1/jobs/$JOB_ID/score" -o score.abc
curl -X POST -H "Authorization: Bearer $YUE2_API_KEY" "http://127.0.0.1:8000/v1/jobs/$JOB_ID/cancel"
```

状态：`queued` → `running` → `succeeded` / `truncated` / `failed` / `cancelled`。
`stage` 为 loading、planning、semantic、synthesis、decode、saving、finished；
`tokens` 为已生成 token 数，不是歌曲完成百分比。时间戳使用 Unix 秒。
结果含音频/可选乐谱地址、采样率、时长、各阶段耗时、实际配置。
`truncated` 明确表示达到 token 上限，音频可下载，但不能当作完整成功。

`cot=off` 不生成乐谱，score 返回 404。翻唱先用 SheetSage2 转谱，再提交 `abc` 文本与
`cot=melody`；此服务不包含原音频转谱。`cfg_scale` 默认沿用上游各模式对应值。
请求体限 256 KiB，文本另有长度验证。若乐谱或生成内容超出模型上下文，任务会失败，不偷偷截短。
队列满返回 429（带 Retry-After），未就绪返回 503。

取消/超时在 token、合成步和解码块之间检查，不能立即抢占已执行的 GPU 运算。
超时默认 1200 秒，从开始执行计时，不含排队；保存期间的取消要等保存结束。

## 运维与恢复

- 一个后台线程独占模型，HTTP 并发不等于 GPU 并发。
- SQLite 保存任务和幂等键；重启继续 queued 任务，原 running 标记 `worker_interrupted`。
- 同一数据目录使用进程锁，禁止多 Uvicorn worker。多 GPU 使用不同 CUDA_VISIBLE_DEVICES、端口和数据目录。
- 推理异常会记录服务端日志并重建模型；HTTP 不暴露原始异常中的内部路径。
- 数据库与 `YUE2_DATA_DIR/artifacts/<id>/` 中的音频、乐谱、生成记录应一起备份。
- 当前不自动清理任务或音频，需按业务保留期清理并监控磁盘；未提供多租户隔离、对象存储或分布式队列。

## 加速配置

默认服务：`torch + resident_models=true + BF16 + 32 步`。
上游每次解码前把主模型移到 CPU，结束再把 VAE 移回 CPU；常驻模式使两者留在 GPU 复用，
减少模型搬运和缓存清理。不减少歌词、token 预算或合成步数。
上游原有 CUDA Graph / FlashAttention 仍保留，不计作本分支新增优化。

服务启动预加载和短预热，将初始化移出正式任务；短预热不覆盖所有形状，上游仍会按生成创建 CUDA Graph。
`YUE2_WARMUP=false` 跳过短预热，预加载保留。
显存不足时设 `YUE2_RESIDENT_MODELS=false` 恢复原策略再测。
`YUE2_MEMORY_BUDGET_GIB=30` 针对 32 GB 卡，内部分配器还会预留余量；其他显存版本需调整。

实验选项需要压测、试听后启用：

- `YUE2_ODE_STEPS=16`：减少合成迭代，可能影响音质；只加速合成阶段，不代表整首快一倍。
- `YUE2_QUANTIZATION=fp8`：上游实验实现，会退出当前 CUDA Graph 路径，可能更慢；默认关闭。
- `YUE2_BACKEND=vllm`：需另装 `.[fast]` 并关闭 resident_models。当前适配器仅单请求，合成时还会关闭
  vLLM 引擎，部分 CFG 设置会回退 torch。基础 Docker 镜像不预装这个可选依赖。

## 真实 GPU 对比

停止占同一 GPU 的其他服务，在主机虚拟环境中运行：

```bash
yue2-benchmark --check
yue2-benchmark --requests examples/benchmark-requests.jsonl \
  --profiles reference resident --warmup 1 --repeats 3 \
  --output outputs/benchmark-5090
```

reference = 上游搬运策略＋原 CUDA Graph；resident = 常驻模式。
两组使用相同请求、种子和默认 32 步；加载和整曲预热单独记录。输出目录必须是新目录。
每个候选保留完整音频及中间结果，report.json 包含：

- 环境、显卡、模型身份、实际配置与实际后端。
- 每次生成/保存耗时、歌曲长度、RTF（生成秒数÷音频秒数）、分阶段时间。
- PyTorch 已分配/保留显存峰值（不等同于整张卡的 NVML 峰值）。
- 失败、截断数量；完整成功任务 p50/p95（最近秩法），不计加载、预热、保存、排队、上传。

三条示例只供冒烟，正式评估应使用 20～30 条实际请求，覆盖中英文、长短歌词、外部乐谱和 CFG，
固定模型版本并重复。比较速度时检查输出时长、截断标记，不能把歌变短算作加速；
再试听歌词准确度、旋律、噪声和结尾。CPU 小模型等价性测试不证明完整模型的 GPU 性能和质量。

## 开发验证

```bash
python -m pip install '.[server,test]'
python -m pytest -q
```

接口测试使用模拟模型；常驻模式测试使用真实小型 VAE 检查输出一致和解码中断。
完整模型与 CUDA 专项测试需要 GPU。代码沿用 Apache 2.0；权重仍受 CC BY-NC 4.0 约束。
