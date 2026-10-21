"""LoRA supervised fine-tuning with transformers.Trainer + peft.

Loss is computed only on assistant (JSON) tokens; prompt tokens are masked with -100.
Usage: python src/train.py --config configs/smollm2_135m_lora.yaml [--max_steps 50 ...]
"""
import argparse
import json
import os
import sys
import time

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

sys.path.insert(0, os.path.dirname(__file__))
from prompting import build_messages, prompt_text  # noqa: E402


def load_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args, rest = ap.parse_known_args()
    cfg = yaml.safe_load(open(args.config))
    for k, v in zip(rest[::2], rest[1::2]):  # simple --key value overrides
        k = k.lstrip("-")
        cfg[k] = type(cfg[k])(v) if k in cfg and cfg[k] is not None and not isinstance(cfg[k], list) else v
    return cfg


class JsonlLogger(TrainerCallback):
    def __init__(self, path):
        self.path = path
        open(path, "w").close()

    def on_log(self, args, state, control, logs=None, **kw):
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": state.global_step, **(logs or {})}) + "\n")


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    torch.set_num_threads(os.cpu_count())
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(ex):
        prompt = prompt_text(tok, ex["notes"])
        full = tok.apply_chat_template(build_messages(ex["notes"], ex["target"]), tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][: cfg["max_length"]]
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels[: len(f_ids)]}

    files = {s: os.path.join(cfg["data_dir"], f"{s}.jsonl") for s in ("train", "validation")}
    ds = load_dataset("json", data_files=files)
    ds = ds.map(encode, remove_columns=ds["train"].column_names)

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (m - len(seq))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    steps_per_epoch = -(-len(ds["train"]) // (cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]))
    total_steps = cfg["max_steps"] if cfg["max_steps"] > 0 else int(steps_per_epoch * cfg["num_train_epochs"])
    warmup_steps = max(1, int(cfg["warmup_ratio"] * total_steps))  # warmup_ratio arg removed in newer transformers
    targs = TrainingArguments(
        output_dir=out, seed=cfg["seed"], num_train_epochs=cfg["num_train_epochs"], max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"], per_device_eval_batch_size=4,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"], learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"], warmup_steps=warmup_steps,
        weight_decay=cfg["weight_decay"], logging_steps=cfg["logging_steps"], eval_strategy="steps",
        eval_steps=cfg["eval_steps"], save_strategy="no", report_to=[], use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0, remove_unused_columns=False)
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["validation"],
                      data_collator=collate, callbacks=[JsonlLogger(os.path.join(out, "train_log.jsonl"))])
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    final_eval = trainer.evaluate()
    model.save_pretrained(os.path.join(out, "adapter"))
    tok.save_pretrained(os.path.join(out, "adapter"))
    summary = {"config": cfg, "train_seconds": round(elapsed, 1), "final_eval": final_eval,
               "global_steps": trainer.state.global_step,
               "device": "cuda" if torch.cuda.is_available() else f"cpu x{os.cpu_count()} threads",
               "torch": torch.__version__}
    json.dump(summary, open(os.path.join(out, "train_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
