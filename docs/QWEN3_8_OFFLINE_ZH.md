# Qwen3.8 FSDP2 LoRA：离线 A800 服务器

目标 A800 不需要访问 GitHub、PyPI 或 Hugging Face。先在联网的兼容 Linux 机器
准备代码、依赖 wheels、完整模型，再通过内网共享盘、跳板机或离线介质传入。
依赖打包机要求 Linux x86_64、Python 3.12、相同或兼容的发行版/glibc，
以及匹配 CUDA 12.8 的编译环境。不要用 macOS/Windows 的 Python 环境打包 Linux wheels。
默认编译目标 `TORCH_CUDA_ARCH_LIST=8.0` 对应 A800。

这里给出 wheelhouse 路径；如果内网已有包含这些依赖的镜像或共享 Python 环境，
可以直接使用，跳过安装。已运行验证的完整容器镜像也可以由联网机器导出后在离线机器导入。

## 1. 联网 Linux 机器准备

```bash
git clone -b codex/qwen38-fsdp2-lora https://github.com/MLcore-Engine/SteptronOss.git
cd SteptronOss
export TORCH_CUDA_ARCH_LIST=8.0
bash tools/install_qwen3_8_fsdp2_env.sh
source .venv-qwen38/bin/activate
QWEN38_BUNDLE=/data/qwen38-offline bash tools/prepare_qwen3_8_offline.sh
```

若打包机没有 GPU，但有兼容的 CUDA Toolkit，可以在安装时设
`QWEN38_REQUIRE_CUDA=0` 跳过最后的 GPU 可见性检查；算子编译仍需要 Toolkit。
依赖打包脚本会冻结当前专用环境全部非 editable 依赖，构建/下载 wheels，并做
`--no-index --only-binary=:all:` 离线解析检查。FlashAttention / causal-conv1d 的 wheel
可能仍需要在联网机器重新编译；仅下载源代码压缩包不够。
打包脚本只归档已提交的仓库代码，不包含模型、数据或未提交编辑。

使用完整独立目录下载模型；可以换成固定的模型 commit：

```bash
hf download Qwen/Qwen3.8-27B --local-dir /data/Qwen3.8-27B
```

完整目录需要 config、tokenizer 文件、所有 safetensors 分片及其 index。
不能只复制某个 Hugging Face snapshot 内指向 cache blobs 的软链接；
使用上述独立目录，或复制时解引用软链接。建议记录模型 commit 并在后续保持权重不变。
如果模型已经在内网完整可用，就不必重新下载。

## 2. 传入服务器

需要传入：

```text
/data/qwen38-offline/
  SteptronOss.tar
  requirements.lock
  SHA256SUMS
  wheels/*.whl
/data/Qwen3.8-27B/      # 单独传输完整模型，体积明显大于代码
/data/train.jsonl      # 你的纯文本 messages 数据
```

若联网机器可以通过内网/跳板连接 A800，可用 rsync；否则使用你已有的共享盘或离线介质。
具体主机和路径按实际环境填写，不要求 A800 出公网：

```bash
rsync -a --info=progress2 /data/qwen38-offline/ infra@A800_HOST:/data/qwen38-offline/
rsync -aL --info=progress2 /data/Qwen3.8-27B/ infra@A800_HOST:/data/Qwen3.8-27B/
```

## 3. 离线服务器安装

先确认操作系统、架构、Python 和可用空间：

```bash
cat /etc/os-release
uname -m
python3.12 --version
nvidia-smi
df -h /data
cd /data/qwen38-offline
sha256sum -c SHA256SUMS
tar -xf SteptronOss.tar
cd SteptronOss
QWEN38_WHEELHOUSE=/data/qwen38-offline/wheels \
QWEN38_OFFLINE_LOCK=/data/qwen38-offline/requirements.lock \
bash tools/install_qwen3_8_fsdp2_env.sh
source .venv-qwen38/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch, flash_attn, fla, causal_conv1d; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
```

安装通过本地 wheelhouse 完成，使用 `--no-index` 禁止查询在线包索引，
并禁止从源码包构建依赖。repo editable 安装使用已带入的 hatchling，关闭 build isolation。
离线机器仍需预先提供 Python 3.12、venv、系统运行库和 NVIDIA 驱动；wheelhouse 不包含这些。
这里没有替你在实际 A800 上构建或安装 wheelhouse，依赖包也尚未实际离线验证。
专用训练环境不安装 vLLM，不能当成整个上游 RL/推理环境使用。

若提示 `No matching distribution`，将缺失包加入联网打包环境重新生成 bundle；
若算子提示 undefined symbol / GLIBC / libcudart 等错误，需要按目标 OS、torch 和 CUDA
重新构建 wheel，不能靠放开 pip 网络选项解决。

## 4. 禁止训练过程联网

```bash
export QWEN38_OFFLINE=1
export QWEN38_BACKEND=fsdp2
export QWEN38_MODEL=/data/Qwen3.8-27B
export QWEN38_DATA=/data/train.jsonl
export QWEN38_GPUS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 先用两卡测试验证通信和续训。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_HUB_KERNELS=0 WANDB_MODE=offline \
torchrun --standalone --nproc-per-node=2 -m pytest -q tests/test_qwen3_8_fsdp2_gpu.py -m node2

# 然后实际 27B 八卡短序列 smoke。
QWEN38_LENGTH=512 QWEN38_BATCH=8 QWEN38_ITERS=3 QWEN38_OUT=/data/qwen38_smoke \
bash tools/run_qwen3_8_lora.sh scheduler_cfg.warmup_schedule=0
```

`QWEN38_OFFLINE=1` 入口会先检查本地 config/tokenizer/模型分片，设置
`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`USE_HUB_KERNELS=0`、`WANDB_MODE=offline`。
本版本默认日志为本地 TensorBoard，不需要 W&B 账号。
`USE_HUB_KERNELS=0` 禁用远程 kernel hub；仍使用已安装的本地 FlashAttention、FLA 和
causal-conv1d。Triton 可能在第一次运行时本地 JIT 编译，这不是公网下载。
这些开关管控对应库，不能代替操作系统的网络隔离策略。

不要离线使用默认的 Hub 模型 ID；必须指定完整本地模型目录。
若直接运行 Python 实验而不使用 shell 入口，上述离线变量也必须在 import Transformers 前设置。

## 5. smoke 通过后验证 16K

```bash
QWEN38_LENGTH=16384 QWEN38_BATCH=16 QWEN38_ITERS=100 QWEN38_OUT=/data/qwen38_16k \
bash tools/run_qwen3_8_lora.sh
```

使用真正接近目标长度的样本测量显存。16K/32K/64K 容量均需在实际 A800 上验证，
离线支持不会改变模型的显存和并行边界。完整参数、恢复和容量测量见
[FSDP2 运行说明](QWEN3_8_FSDP2_LORA_ZH.md)。
