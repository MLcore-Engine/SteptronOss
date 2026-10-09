#!/usr/bin/env bash
# Dedicated training environment: avoid upstream vLLM's torch dependency.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
qwen38_env="${QWEN38_ENV:-.venv-qwen38}"
qwen38_pip_source=()
if [[ -n "${QWEN38_WHEELHOUSE:-}" ]]; then
    [[ -d "$QWEN38_WHEELHOUSE" ]] || { printf 'Wheelhouse does not exist: %s\n' "$QWEN38_WHEELHOUSE" >&2; exit 2; }
    export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
    qwen38_pip_source=(--no-index --find-links "$QWEN38_WHEELHOUSE" --only-binary=:all:)
fi
"${QWEN38_PYTHON:-python3.12}" -m venv "$qwen38_env"
qwen38_python="$qwen38_env/bin/python"
if [[ -n "${QWEN38_WHEELHOUSE:-}" ]]; then
    : "${QWEN38_OFFLINE_LOCK:?Set QWEN38_OFFLINE_LOCK to the lock file exported with this wheelhouse}"
    [[ -f "$QWEN38_OFFLINE_LOCK" ]] || { printf 'Offline lock does not exist\n' >&2; exit 2; }
    # The lock contains bootstrap packages, torch, every transitive dependency,
    # and optional kernels. Install all of them from local wheels in one pass.
    "$qwen38_python" -m pip install "${qwen38_pip_source[@]}" -r "$QWEN38_OFFLINE_LOCK"
else
    "$qwen38_python" -m pip install --upgrade pip setuptools wheel packaging ninja hatchling
    "$qwen38_python" -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
fi
# Constrain later installs so optional kernels cannot silently upgrade torch.
qwen38_constraints="$qwen38_env/qwen38-constraints.txt"
printf 'torch==2.11.0\ntransformers==5.19.0\npeft==0.21.2\n' > "$qwen38_constraints"
if [[ -z "${QWEN38_WHEELHOUSE:-}" ]]; then
    "$qwen38_python" -m pip install -c "$qwen38_constraints" -r requirements/qwen38-fsdp2-runtime.txt
fi
"$qwen38_python" -m pip install -e . --no-deps --no-build-isolation
if [[ -z "${QWEN38_WHEELHOUSE:-}" && "${QWEN38_INSTALL_KERNELS:-1}" == 1 ]]; then
    "$qwen38_python" -m pip install -c "$qwen38_constraints" \
        flash-attn flash-linear-attention causal-conv1d --no-build-isolation
fi
"$qwen38_python" - <<'PY'
import os
import torch, transformers, peft
print("torch", torch.__version__, "CUDA", torch.version.cuda, "transformers", transformers.__version__, "peft", peft.__version__)
if os.environ.get("QWEN38_REQUIRE_CUDA", "1") == "1":
    assert torch.cuda.is_available(), "CUDA is unavailable"
PY
printf 'Activate with: source %s/bin/activate\n' "$qwen38_env"
