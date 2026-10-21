PY ?= python
CONFIG ?= configs/smollm2_135m_lora.yaml
ADAPTER ?= outputs/smollm2-135m-pm-status-lora/adapter

.PHONY: install data train eval demo all clean

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

data:
	$(PY) src/generate_dataset.py --n 480 --seed 42 --out data

train:
	$(PY) src/train.py --config $(CONFIG)

eval:
	$(PY) src/evaluate.py --adapter $(ADAPTER) --out_dir results

demo:
	$(PY) src/demo.py --adapter $(ADAPTER)

all: data train eval

clean:
	rm -rf outputs results/predictions.jsonl
