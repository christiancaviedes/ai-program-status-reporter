#!/usr/bin/env bash
# End-to-end reproduction: data -> LoRA training -> base-vs-finetuned eval.
# Usage: ./run.sh [config]   (default: configs/smollm2_135m_lora.yaml)
set -euo pipefail
CONFIG="${1:-configs/smollm2_135m_lora.yaml}"
OUT_DIR=$(python -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
BASE=$(python -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['model_name'])")
TAG=$(basename "$OUT_DIR" | sed 's/-pm-status-lora//')
python src/generate_dataset.py --n 480 --seed 42 --out data
python src/train.py --config "$CONFIG"
python src/evaluate.py --base_model "$BASE" --adapter "$OUT_DIR/adapter" --out_dir "results/$TAG"
