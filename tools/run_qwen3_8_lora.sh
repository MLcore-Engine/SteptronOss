#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

: "${QWEN38_DATA:?Set QWEN38_DATA to your messages JSONL file}"
QWEN38_MODEL="${QWEN38_MODEL:-Qwen/Qwen3.8-27B}"
if [[ "${QWEN38_OFFLINE:-0}" == 1 ]]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_HUB_KERNELS=0 WANDB_MODE=offline
    [[ -d "$QWEN38_MODEL" ]] || { printf 'Offline mode requires QWEN38_MODEL to be a local model directory\n' >&2; exit 2; }
    python - "$QWEN38_MODEL" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
required = [root / 'config.json', root / 'tokenizer.json', root / 'tokenizer_config.json']
index = root / 'model.safetensors.index.json'
if index.is_file():
    required += [root / shard for shard in set(json.loads(index.read_text())['weight_map'].values())]
else:
    required.append(root / 'model.safetensors')
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f'Incomplete local model: {missing}')
PY
fi
QWEN38_GPUS="${QWEN38_GPUS:-8}"
QWEN38_LENGTH="${QWEN38_LENGTH:-2048}"
QWEN38_BATCH="${QWEN38_BATCH:-32}"
QWEN38_OUT="${QWEN38_OUT:-./qwen38_output}"
QWEN38_BACKEND="${QWEN38_BACKEND:-dp}"
case "$QWEN38_BACKEND" in
    dp) qwen38_experiment="playground/sft/qwen3_8/qwen3_8_27b_lora.py" ;;
    fsdp2) qwen38_experiment="playground/sft/qwen3_8/qwen3_8_27b_fsdp2_lora.py" ;;
    *) printf 'QWEN38_BACKEND must be dp or fsdp2\n' >&2; exit 2 ;;
esac

prep_args=(--model "$QWEN38_MODEL" --input "$QWEN38_DATA"
           --output "$QWEN38_OUT/data/train.tokens.jsonl" --max-length "$QWEN38_LENGTH")
train_args=("model_cfg.model_path=$QWEN38_MODEL"
            "data_cfg.tokenized_path=$QWEN38_OUT/data/train.tokens.jsonl"
            "resource_cfg.gpu=$QWEN38_GPUS"
            "trainer_cfg.global_seq_length=$QWEN38_LENGTH"
            "trainer_cfg.global_batch_size=$QWEN38_BATCH"
            "checkpoint_cfg.save_dir=$QWEN38_OUT/checkpoints"
            "log_dir=$QWEN38_OUT/logs")
if [[ -n "${QWEN38_REVISION:-}" ]]; then
    prep_args+=(--revision "$QWEN38_REVISION")
    train_args+=("model_cfg.revision=$QWEN38_REVISION")
fi
if [[ -n "${QWEN38_ITERS:-}" ]]; then
    train_args+=("trainer_cfg.train_iters=$QWEN38_ITERS")
fi
# Preparation reports the tokenizer's pad ID; read it rather than assume it.
python tools/prepare_qwen3_8_sft.py "${prep_args[@]}"
qwen38_pad_id="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["pad_token_id"])' "$QWEN38_OUT/data/train.tokens.jsonl.meta.json")"
train_args+=("data_cfg.pad_token_id=$qwen38_pad_id")
torchrun --standalone --nproc-per-node="$QWEN38_GPUS" \
    "$qwen38_experiment" "${train_args[@]}" "$@"
