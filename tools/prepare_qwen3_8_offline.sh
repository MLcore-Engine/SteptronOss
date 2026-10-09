#!/usr/bin/env bash
# Run in the dedicated ONLINE Linux x86_64 Python 3.12 training environment.
# Models and training data are transferred separately, not duplicated here.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${QWEN38_BUNDLE:?Set QWEN38_BUNDLE to an output directory outside the repository}"
qwen38_python="${QWEN38_PYTHON:-python}"
"$qwen38_python" - <<'PY'
import importlib, platform, sys
import torch, transformers, peft
if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 12):
    raise SystemExit('Prepare this bundle on Linux x86_64 with Python 3.12, not macOS or Windows')
if torch.__version__ != '2.11.0+cu128' or transformers.__version__ != '5.19.0' or peft.__version__ != '0.21.2':
    raise SystemExit('Activate the dedicated torch 2.11.0+cu128 / transformers 5.19.0 / peft 0.21.2 environment')
for name in ('hatchling', 'flash_attn', 'fla', 'causal_conv1d'):
    importlib.import_module(name)
print('Packaging', platform.platform(), 'glibc', platform.libc_ver(), 'Python', sys.version.split()[0])
PY
mkdir -p "$QWEN38_BUNDLE/wheels"
"$qwen38_python" -m pip freeze --all --exclude-editable > "$QWEN38_BUNDLE/requirements.lock"
# A freeze containing local/direct URL dependencies is not portable.
"$qwen38_python" - "$QWEN38_BUNDLE/requirements.lock" <<'PY'
from pathlib import Path
import sys
lines = Path(sys.argv[1]).read_text().splitlines()
if any(line.strip() and not line.startswith('#') and '==' not in line for line in lines):
    raise SystemExit('Lock contains an unpinned/local/URL dependency; use a clean environment with registry packages')
PY
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
"$qwen38_python" -m pip wheel --no-build-isolation \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --wheel-dir "$QWEN38_BUNDLE/wheels" -r "$QWEN38_BUNDLE/requirements.lock"
# Confirm offline resolution before copying anything to the A800 host.
"$qwen38_python" -m pip install --dry-run --ignore-installed --no-index \
    --only-binary=:all: --find-links "$QWEN38_BUNDLE/wheels" -r "$QWEN38_BUNDLE/requirements.lock"
git archive --format=tar --prefix=SteptronOss/ HEAD > "$QWEN38_BUNDLE/SteptronOss.tar"
"$qwen38_python" - "$QWEN38_BUNDLE" <<'PY'
import hashlib
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
paths = [root / 'requirements.lock', root / 'SteptronOss.tar', *sorted((root / 'wheels').glob('*.whl'))]
with (root / 'SHA256SUMS').open('w') as out:
    for path in paths:
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        out.write(f'{digest}  {path.relative_to(root)}\n')
PY
printf 'Bundle ready: %s\nTransfer model and data separately. Only committed code was archived.\n' "$QWEN38_BUNDLE"
