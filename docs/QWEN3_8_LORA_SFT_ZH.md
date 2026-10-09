# 在 SteptronOSS 中训练 Qwen3.8-27B LoRA SFT

本实现基于上游 `dev` 提交 `5169489282b14e83efd581225540ad9f2ca2c5c9`。

本文说明 DP 入口。单机 8 × A800 的新增 FSDP2 分片方案见
[FSDP2 LoRA 运行说明](QWEN3_8_FSDP2_LORA_ZH.md)，通过 `QWEN38_BACKEND=fsdp2` 选择。

## 实现范围

训练主控为 SteptronOSS：实验配置、`DecoderPretrainTrainer.train_loop/train_step`、
`FWBWScheduler`、`AccInFP32GradientManager`、DP 梯度 all-reduce、学习率调度和
`Checkpointer` 均沿用原框架。模型的 Qwen3.5/3.8 混合层由 Transformers 实现，
LoRA 注入与标准 adapter 导出由 PEFT 实现。没有调用 Transformers Trainer、
ms-swift 或 LLaMA-Factory。

| 项目 | 当前支持 |
|---|---|
| Qwen3.8-27B 纯文本 LoRA SFT | 已实现接入，27B GPU 实跑待验证 |
| 并行 | DP；每张卡有完整基础模型，native all-reduce 同步 LoRA 梯度 |
| LoRA 目标 | 文本 decoder 中所有 Linear，包含 Gated DeltaNet 和全注意力层 |
| 基础模型、视觉编码器、LM head | LoRA 模式均冻结 |
| 标签 | 只监督 assistant；包含推理内容和结束标记；HF 内部移位一次 |
| 多轮对话、推理内容 | 支持；历史 reasoning 保留以维持模板前缀一致 |
| checkpoint | adapter + optimizer + scheduler + data + RNG；自动恢复 |
| 导出 | 各 checkpoint 的 `hf_export/` 为标准 PEFT adapter |
| TP / PP / CP / EP | 此接入尚未实现，配置检查会拒绝 |
| 图片、视频训练 | 尚未实现；数据预处理会拒绝 |
| QLoRA / 4bit | 尚未实现 |
| 全参数文本 SFT | `model_cfg.tuner=full` 接口已实现，仅小模型验证；27B DP 内存需求很高 |

这是复用模型算子的 SteptronOSS 接入版本。若需要把 Qwen3.8 层重写成原生
TP/PP 可切分模块，需要进一步实现 GDN 的并行与权重转换。

## 环境

建议单独的 Python 3.12 环境；按你的驱动/CUDA 安装匹配的 PyTorch。
上游要求 `torch>=2.9`，不能直接沿用较老的 torch 环境。

在仓库根目录执行：

```bash
uv sync --extra qwen38-hf
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

这会安装 SteptronOSS 的原有依赖及模型接入所需的 Transformers/PEFT。
原仓库含 vLLM 依赖，因此环境解析可能耗时。

Gated DeltaNet 的快路径建议安装 FLA 和 causal-conv1d：

```bash
uv pip install 'flash-linear-attention>=0.4.2' --no-build-isolation
uv pip install causal-conv1d --no-build-isolation
```

它们必须匹配本机 CUDA、PyTorch 和 GPU 架构。未安装时小模型可运行参考实现，
但 27B 的训练速度和显存可能明显变差。`attn_implementation=sdpa` 只决定全注意力
层，不能替代 GDN 内核。此训练入口没有使用 RL/生成服务，也无需启动 vLLM。

当前验证环境为 PyTorch 2.11.0+cpu、Transformers 5.19.0、PEFT 0.21.2。
这只是本地 CPU 验证组合，不是已经验证过的 CUDA 安装组合。

## 数据格式

每行一个对象，最后一条必须是 assistant，content 使用字符串：

```json
{"messages":[{"role":"user","content":"Pod 一直 Pending，首先检查什么？"},{"role":"assistant","content":"先检查 kubectl describe pod 中的调度事件，再核对资源、节点选择条件和 PVC 状态。"}]}
{"messages":[{"role":"user","content":"什么是流水线并行？"},{"role":"assistant","content":"把模型的连续层分到不同 GPU，并通过多个 micro batch 让各阶段重叠工作。"}]}
```

多轮对话也可以。已有推理过程可放在 assistant 的 `reasoning_content` 字段，
按照官方模板序列化。不要把推理标签字符串混进 content 后再次加一层标签。

默认预处理以非思考模式序列化，空 `<think>` 块也作为 assistant 输出进行训练。
若训练 reasoning，单独调用预处理时加入 `--enable-thinking`，并合理设置
`--reasoning-effort low/medium/xhigh`。

默认 global batch 为 32，至少需要 32 条有效数据。loader 每个 epoch 会丢弃
不足一个 global batch 的尾部，然后在需要继续训练时以新 shuffle 开始下一轮。
不能把这里的两条示例直接拿去跑默认 batch。此版本将预处理 token 数据载入内存，
适合中小规模微调；超大数据集后续应接入原仓库 mmap/compiled dataset 管线。

## 最短启动路径

如果使用 8 张 80GB 级 GPU，可以先以 512 tokens、micro batch 1、global batch 8
做 3 步 smoke。是否适合你的设备仍取决于 GPU 显存、算子版本及实际峰值。

```bash
QWEN38_MODEL=/models/Qwen3.8-27B \
QWEN38_DATA=/data/train.jsonl \
QWEN38_GPUS=8 \
QWEN38_LENGTH=512 \
QWEN38_BATCH=8 \
QWEN38_ITERS=3 \
QWEN38_OUT=./qwen38_smoke \
bash tools/run_qwen3_8_lora.sh scheduler_cfg.warmup_schedule=0
```

`QWEN38_MODEL` 可以是 Hub ID，也可以是包含 config、tokenizer 和 safetensors 的
完整本地目录。推荐固定 Hub commit（`QWEN38_REVISION`）或使用不再变动的本地模型。
不要将 AWQ/GGUF/FP8 推理量化权重作为这里的 BF16 基础模型。

脚本先预处理，再用 torchrun 启动 SteptronOSS 的实验；最后的参数是原框架
`key=value` 覆盖。global batch 必须整除 GPU 数 × micro batch。
比如 8 卡、micro batch 1、global batch 32，每卡每次更新累计 4 个 micro batch。

正式训练可使用独立输出目录：

```bash
QWEN38_MODEL=/models/Qwen3.8-27B \
QWEN38_DATA=/data/train.jsonl \
QWEN38_GPUS=8 \
QWEN38_LENGTH=2048 \
QWEN38_BATCH=32 \
QWEN38_OUT=./qwen38_sft \
bash tools/run_qwen3_8_lora.sh
```

不设置 `QWEN38_ITERS` 时，native trainer 自动按一个完整数据 epoch 计算迭代数。
设置迭代数超过一个 epoch 会继续循环并重新 shuffle。LoRA rank、learning rate 等：

```bash
bash tools/run_qwen3_8_lora.sh \
  model_cfg.lora_rank=32 \
  model_cfg.lora_alpha=64 \
  scheduler_cfg.lr=5e-5
```

初始化环境变量后再运行上面的命令；也可直接将这些覆盖加在正式训练命令末尾。

## 手动预处理与配置检查

```bash
python tools/prepare_qwen3_8_sft.py \
  --model /models/Qwen3.8-27B \
  --input /data/train.jsonl \
  --output ./data/qwen3_8_train.tokens.jsonl \
  --max-length 2048

PYTHONPATH=. cfshow playground/sft/qwen3_8/qwen3_8_27b_lora.py
```

预处理会生成 token JSONL 和 `.meta.json`。训练验证模型路径、revision、长度上限、
pad token 及 token 文件 SHA256，避免误用其他 tokenizer 或已经变动的数据。
手动运行 torchrun 时，指定与预处理一致的 `model_cfg.model_path` 和
`data_cfg.tokenized_path`；pad_token_id 使用预处理结果中打印的值。
超过长度的记录采用右侧截断；若截断后没有 assistant target，则报错而不静默丢弃。

## 内存与并行

27B × BF16 的语言模型权重约为 54GB（十进制，约 50.3GiB），视觉权重、fp32
状态、adapter、激活、loss/logits 和算子工作区还会额外占用显存。
因此 DP8 不会把基础权重分摊成每卡 6.75GB；每卡都有完整权重。

80GB 级 GPU 建议从 512 tokens / micro batch 1 验证，记录峰值后再提高到 2K/4K。
40GB 卡不能直接使用这条 BF16 DP 路径；增加 DP 卡数不会解决每卡放不下基础模型。
本版本没有给出未实测的吞吐或保证显存数字。

不要在 8×80GB 上将 `tuner=full` 当作已可运行的 27B 全参方案。
全参训练的每卡梯度、主权重、Adam 状态远大于 LoRA，此版本尚无 TP/PP/FSDP
来分摊它们。`full` 仅是代码路径和小模型功能验证。

## checkpoint 与恢复

输出在 `<QWEN38_OUT>/checkpoints/qwen3_8_27b_lora/`，默认每 100 步保存，
训练完成也会保存最后一次更新。`latest_ckpt` 指向最近的完整 native checkpoint。
其中 `itN` 的 N 是从 0 开始的最后已完成 update index。

恢复时沿用同一输出目录、基础模型和数据即可。`train_iters` 是总训练目标，
不是恢复后额外运行的步数。若已跑完 3 步，把目标改成 10 会再运行 7 步。

GPU 数、global/micro batch、token 长度、LoRA 配置、数据 hash 改变时，恢复会拒绝。
LoRA adapter 只有低秩增量，推理时仍需要原始基础模型。

`hf_export/` 包含标准 `adapter_config.json` 与 `adapter_model.safetensors`；
通过 PEFT 的 `PeftModel.from_pretrained(base_model, adapter_dir)` 加载。
使用基础模型目录的 AutoTokenizer 和 Qwen3.8 chat template。若推理服务不支持
Qwen3.8 的动态 LoRA，可离线执行 PEFT `merge_and_unload()` 后保存完整 HF 模型，
再部署合并权重；合并需要足够的 CPU/GPU 内存。

## 验证

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 python -m pytest \
  tests/test_qwen3_8_sft.py tests/test_qwen3_8_sft_gpu.py -q

PYTHONPATH=. OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 \
  -m pytest tests/test_qwen3_8_sft_gpu.py -m node2 -q
```

CPU 测试使用真实 Transformers Qwen3.5 混合结构（3 个 GDN + 1 个全注意力层），
验证 HF 基础权重加载、标签位移、loss、LoRA 梯度与更新、冻结权重、adapter 恢复/
导出、DP 数据切分、原生 FWBWScheduler 的梯度累积和 Checkpointer 的状态收集。
另以官方 Qwen3.8-27B tokenizer 实际检查了中文多轮 assistant mask。

GPU 测试会真正运行 native trainer、原生梯度管理器和 checkpoint，再恢复继续训练；
当前环境没有 GPU，这项尚未执行，不能把 CPU 通过等同于 27B 训练已跑通。

定位文件：

- 模型桥接：`steptronoss/model/qwen3_8_hf.py`
- 数据掩码与 loader：`steptronoss/data/qwen3_8_sft.py`
- native trainer 扩展：`steptronoss/core/trainers/qwen3_8_sft_trainer.py`
- 实验：`playground/sft/qwen3_8/qwen3_8_27b_lora.py`
- 数据预处理与启动：`tools/prepare_qwen3_8_sft.py`、`tools/run_qwen3_8_lora.sh`

参考：

- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/docs/transformers/main/model_doc/qwen3_5
- https://huggingface.co/docs/peft/main/en/package_reference/lora
