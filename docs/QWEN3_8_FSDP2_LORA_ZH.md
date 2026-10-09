# Qwen3.8-27B：单机 8 × A800 80GB 的 FSDP2 LoRA SFT

基于上游 dev `5169489282b14e83efd581225540ad9f2ca2c5c9`。
这是新增的 FSDP2 后端；上游原生并行不是 FSDP2。

## 实现与边界

训练入口是 `playground/sft/qwen3_8/qwen3_8_27b_fsdp2_lora.py`。
继续使用 SteptronOSS 的配置、训练循环、FW/BW scheduler、日志和学习率调度；
模型算子由 Transformers 实现，LoRA 由 PEFT 注入，分片和梯度通信由 PyTorch FSDP2 负责。
没有另外启动 HF Trainer 或 vLLM 服务。

| 项目 | 本分支行为 |
|---|---|
| 范围 | Qwen3.8-27B 原始 BF16 权重，纯文本 LoRA SFT |
| 模型架构 | Transformers `qwen3_5` 混合 Gated DeltaNet / 全注意力 |
| 分片 | decoder block 的 BF16 基础权重与 FP32 adapter 分组；根组含 embedding、head、冻结视觉塔 |
| 目标 | 文本 decoder 所有 Linear；基础参数、视觉塔、LM head 冻结 |
| 显存优化 | 基础权重分片、非重入激活重计算、FlashAttention 2、分块 LM head loss |
| 初始化 | rank 0 加载 CPU 基础权重，其余 rank 以 meta 初始化，再广播分片 |
| 检查点 | 每 rank 保存 adapter 分片、Adam、scheduler、数据游标、RNG；完成 manifest 后发布 latest |
| 导出 | 全 rank 参与，仅聚合 adapter，由 rank 0 写标准 PEFT `hf_export/` |
| 限制 | 同 world size 恢复；同步写共享本地文件系统；不支持 TP/PP/CP、QLoRA、多模态或全参 FSDP2 |

FSDP2 本身是分片数据并行：各 rank 训练不同样本，共享逻辑模型。
它不会把一条样本的序列分到八卡，因此长上下文激活仍是每卡的主要限制。
不能再叠加 SteptronOSS 原生 ZeRO-1、原生梯度 buffer/all-reduce 或优化器 offload。

分块 loss 是 PyTorch 的精确交叉熵参考实现，**尚未实现融合 Triton loss**。
它只为有监督的 token 投影 logits，前向与反向均按 chunk 重算，避免完整
`sequence × vocabulary` logits，但仍保留 decoder 隐状态和激活。只支持冻结 head、
一阶训练梯度；开启 labels 的训练路径不返回 logits。检查点和导出只能在训练步之间调用。

## 环境安装

针对用户提供的 570.124.06 驱动，选用 PyTorch 2.11 的 CUDA 12.8 wheel。
`nvidia-smi` 的 CUDA Version 表示驱动支持能力，不表示已安装 `nvcc`。
编译 FlashAttention / causal-conv1d 时还需要匹配的 CUDA Toolkit、编译器和 ninja。
先检查 `nvcc --version` 和 `CUDA_HOME`，不要将另一台机器的 Toolkit 路径直接复制过来。

```bash
bash tools/install_qwen3_8_fsdp2_env.sh
source .venv-qwen38/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch, flash_attn, fla, causal_conv1d; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
nvidia-smi topo -m
```

脚本创建独立环境并固定 torch/Transformers/PEFT，避免上游 vLLM 依赖改变 torch。
它不是整个上游 RL/推理环境。可设 `QWEN38_ENV` 更换环境目录、
`QWEN38_PYTHON` 更换 Python 3.12 可执行文件。
`QWEN38_INSTALL_KERNELS=0` 仅跳过可选算子安装；默认实验仍要求 FlashAttention 2。
小模型 GPU 测试显式使用 SDPA。生产 27B 的 GDN 快路径还需要 FLA 与 causal-conv1d，
SDPA/FlashAttention 只管理全注意力层，不能替代 GDN 算子。
安装脚本尚未在 A800 主机运行，编译和 CUDA ABI 是否匹配需要现场验证。

rank 0 需要容纳完整 CPU 基础权重，另有加载、token 数据和 Python 开销。
主机 RAM 建议至少 128GB、有条件采用 256GB；这是容量规划，不是实测峰值。
本版本将 token JSONL 加载到每个 rank 的内存，超大数据集需要另外接入 mmap 数据管线。
冻结视觉塔仍加载并分片，纯文本训练不会执行视觉前向。

## 先验证分布式训练与恢复

```bash
python -m pytest -q tests/test_qwen3_8_sft.py tests/test_qwen3_8_fsdp2.py
torchrun --standalone --nproc-per-node=2 -m pytest -q tests/test_qwen3_8_fsdp2_gpu.py -m node2
WORLD_SIZE=8 cfshow playground/sft/qwen3_8/qwen3_8_27b_fsdp2_lora.py
WORLD_SIZE=8 python -c 'from playground.sft.qwen3_8.qwen3_8_27b_fsdp2_lora import Exp; Exp().sanity_check()'
```

两卡测试通过真实 trainer，对比连续训练 4 步和训练 2 步后恢复到 4 步，
验证 adapter、Adam、数据游标、导出及 rank 0/meta 初始化。
通过小模型测试后，再进行实际 27B smoke。

## 数据与八卡 smoke

每行一个纯文本 messages 对象，最后一条为 assistant：

```json
{"messages":[{"role":"user","content":"如何检查 Pod Pending？"},{"role":"assistant","content":"先检查调度事件，再核对资源、节点约束和存储。"}]}
```

监督 assistant 输出，mask prompt 和 padding，labels 在模型内只移位一次。
多轮及 `reasoning_content` 字段见 [数据说明](QWEN3_8_LORA_SFT_ZH.md)。
至少需要一个完整 global batch 的有效样本。超过上限会右截断；截断后无监督 target 会报错。

```bash
export QWEN38_BACKEND=fsdp2
export QWEN38_MODEL=/models/Qwen3.8-27B
export QWEN38_DATA=/data/train.jsonl
export QWEN38_GPUS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QWEN38_LENGTH=512 QWEN38_BATCH=8 QWEN38_ITERS=3 QWEN38_OUT=./qwen38_fsdp2_smoke \
bash tools/run_qwen3_8_lora.sh scheduler_cfg.warmup_schedule=0
```

模型目录须包含原始 config、tokenizer 和 safetensors；不能使用 AWQ、GGUF 或 FP8 推理权重。
Hub 模型可通过 `QWEN38_REVISION` 固定 commit。本地模型目录在续训时也必须保持内容不变。
微批量默认为 1；global batch 必须整除 GPU 数 × micro batch。

## 16K 起步，逐级测量上限

```bash
QWEN38_LENGTH=16384 QWEN38_BATCH=16 QWEN38_ITERS=100 QWEN38_OUT=./qwen38_fsdp2_16k \
bash tools/run_qwen3_8_lora.sh
```

这是默认起步配置：BF16 base、FP32 LoRA rank 16 / alpha 32、micro batch 1、
每卡累计 2 个 micro batch、激活重计算、FlashAttention 2、loss chunk 128。
先观察各 rank 输出的 `FSDP2 peak memory`、loss、梯度范数和步耗时。
峰值包含 PyTorch allocated/reserved；还需用 nvidia-smi 检查 NCCL 等额外开销。

验证 32K 时用新输出目录、新预处理上限，并使用真正接近 32K 的训练样本：

```bash
QWEN38_LENGTH=32768 QWEN38_BATCH=8 QWEN38_ITERS=10 QWEN38_OUT=./qwen38_fsdp2_32k_probe \
bash tools/run_qwen3_8_lora.sh scheduler_cfg.warmup_schedule=0
```

loader 按实际样本长度 padding；设置上限 32K 而只喂 1K 样本，无法验证 32K 容量。
global batch 16 改为 8 主要减少梯度累计和步耗时，**不会降低单个 micro batch 的激活峰值**。
若 logits 临时内存占比较高，可设 `model_cfg.loss_chunk_size=64`，代价是更多投影调用。
若 decoder 激活超限，需缩短序列、减少 LoRA 目标，或另行开发激活 offload / CP；
不能靠继续减少已经为 1 的 micro batch 解决。

**最长可训练上下文尚未实测。** 16K 是起步目标，32K 是优先测试目标，64K 只是后续候选；
不能由 8 × 80GB 的总显存直接得出 64K/128K 可运行。
上限还取决于模型原生上下文配置、真实样本长度、GDN 内核、重计算中间张量及八卡通信。
容量探测至少应包含若干完整前向、反向、优化器步和一次检查点保存。

## 保存与恢复

默认每 100 步及训练结束保存一次：

```text
<OUT>/checkpoints/<exp_name>/
  latest_ckpt
  it99/
    manifest.json
    rank0.pt ... rank7.pt
    hf_export/adapter_config.json
    hf_export/adapter_model.safetensors
```

步编号从 0 开始；it99 表示第 100 次更新完成。
再次以相同输出目录启动会读取 latest 并自动恢复；显式恢复可使用
`checkpoint_cfg.load_path=/path/to/it99`。
恢复时保持 world size、基础权重、数据 hash、序列上限、batch、LoRA、loss 和算子配置一致。
允许增加 `QWEN38_ITERS`，但若要求与连续训练一致，初始运行就应显式固定
`scheduler_cfg.total_schedule` 为最终训练计划长度。
改变序列长度或数据属于新训练，使用新输出目录；本版本不实现 world size 重分片续训。
续训文件不会复制冻结的 27B 权重，因此基础模型目录不可删除。
只有各 rank 完成文件写入、adapter 导出后才更新 latest。

## 已完成的验证

- PyTorch 2.11.0+cpu、Transformers 5.19.0、PEFT 0.21.2：20 个 CPU 测试通过。
- 覆盖真实小型 3 层 GDN + 1 层全注意力模型、LoRA 更新、冻结参数、标签与数据恢复。
- 分块 loss 在 FP32/BF16 下对比完整交叉熵与隐藏层梯度，且前后向投影行数受 chunk 限制。
- 单进程 FSDP2 测试使用 fake process group，真实执行原生 FW/BW scheduler 的微批量累计、
  FSDP2 hooks、DTensor、Adam、导出和恢复；
  **不验证多卡通信**。两项 GPU 测试在本开发环境跳过。
- 配置 sanity、cfshow 及配置差异检查、相关代码 ruff/mypy、shell 语法检查通过。
- 尚未在用户的 A800 上实跑 27B，尚无 16K/32K/64K 峰值或吞吐测量。
