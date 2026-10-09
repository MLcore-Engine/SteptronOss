#!/usr/bin/env bash
# Dedicated training environment: avoid upstream vLLM's torch dependency.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
qwen38_env="${QWEN38_ENV:-.venv-qwen38}"
"${QWEN38_PYTHON:-python3.12}" -m venv "$qwen38_env"
qwen38_python="$qwen38_env/bin/python"
"$qwen38_python" -m pip install --upgrade pip setuptools wheel packaging ninja
"$qwen38_python" -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
# Constrain later installs so optional kernels cannot silently upgrade torch.
qwen38_constraints="$qwen38_env/qwen38-constraints.txt"
printf 'torch==2.11.0\ntransformers==5.19.0\npeft==0.21.2\n' > "$qwen38_constraints"
"$qwen38_python" -m pip install -c "$qwen38_constraints" \
    'transformers==5.19.0' 'peft==0.21.2' 'accelerate>=1.0' \
    'configurize==0.2.0' aiohttp aioredis emoji boto3 diskcache einops fastapi \
    hjson ijson loguru megfile msgpack nltk numpy pyyaml redis safetensors \
    syllapy tabulate tqdm uvicorn wandb 'httpx[socks]' tensorboard asciichartpy \
    debugpy pytest 'mypy==1.19.0' ruff
"$qwen38_python" -m pip install -e . --no-deps
if [[ "${QWEN38_INSTALL_KERNELS:-1}" == 1 ]]; then
    "$qwen38_python" -m pip install -c "$qwen38_constraints" \
        flash-attn flash-linear-attention causal-conv1d --no-build-isolation
fi
"$qwen38_python" -c 'import torch, transformers, peft; print("torch", torch.__version__, "CUDA", torch.version.cuda, "transformers", transformers.__version__, "peft", peft.__version__); assert torch.cuda.is_available(), "CUDA is unavailable"'
printf 'Activate with: source %s/bin/activate\n' "$qwen38_env"
