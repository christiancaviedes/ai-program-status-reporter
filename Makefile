PY ?= python
CONFIG ?= configs/smollm2_135m_lora.yaml
BASE ?= HuggingFaceTB/SmolLM2-135M-Instruct
ADAPTER ?= outputs/smollm2-135m-pm-status-lora/adapter
RESULTS ?= results/smollm2-135m

.PHONY: install data train eval demo all clean

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

data:
	$(PY) src/generate_dataset.py --n 480 --seed 42 --out data

train:
	$(PY) src/train.py --config $(CONFIG)

eval:
	$(PY) src/evaluate.py --base_model $(BASE) --adapter $(ADAPTER) --out_dir $(RESULTS)

demo:
	$(PY) src/demo.py --base_model $(BASE) --adapter $(ADAPTER)

all: data train eval

clean:
	rm -rf outputs
