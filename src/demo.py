"""Minimal demo: turn messy notes into a JSON status report.

CLI:    python src/demo.py --adapter outputs/smollm2-135m-pm-status-lora/adapter --notes_file my_notes.txt
        echo "notes..." | python src/demo.py --adapter ...
Gradio: python src/demo.py --adapter ... --gradio      (requires `pip install gradio`)
"""
import argparse
import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import extract_json  # noqa: E402
from prompting import prompt_text  # noqa: E402

EXAMPLE = """Falcon weekly sync
- design review done ✅ (Mar 3)
* load testing tgt apr 20, on schedule
- p99 latency too high in load test (critical) Wei owns -> add caching layer
AI: Grace to update RAID log
need decision - agree on cutover date"""


def load(base, adapter):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return tok, model.eval()


@torch.no_grad()
def run(tok, model, notes, max_new_tokens=448):
    enc = tok(prompt_text(tok, notes), return_tensors="pt", add_special_tokens=False)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    obj = extract_json(text)
    return json.dumps(obj, indent=2) if obj is not None else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--adapter", default="outputs/smollm2-135m-pm-status-lora/adapter")
    ap.add_argument("--notes_file")
    ap.add_argument("--gradio", action="store_true")
    a = ap.parse_args()
    tok, model = load(a.base_model, a.adapter)
    if a.gradio:
        import gradio as gr
        gr.Interface(fn=lambda n: run(tok, model, n), inputs=gr.Textbox(lines=12, value=EXAMPLE, label="Program notes"),
                     outputs=gr.Code(language="json", label="Status report"),
                     title="Program notes → JSON status report (SmolLM2-135M + LoRA, synthetic-data demo)").launch()
        return
    notes = open(a.notes_file).read() if a.notes_file else (sys.stdin.read() if not sys.stdin.isatty() else EXAMPLE)
    print(run(tok, model, notes))


if __name__ == "__main__":
    main()
