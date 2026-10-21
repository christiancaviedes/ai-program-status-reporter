"""Evaluate base vs LoRA fine-tuned model on the held-out test split.

Metrics (all computed per example, then aggregated):
  json_valid_rate     - output parses as a JSON object (after stripping ``` fences / surrounding text)
  strict_json_rate    - the raw output (whitespace-stripped) parses as JSON with no extraction
  schema_coverage     - mean fraction of the 7 required top-level keys present (0 if invalid JSON)
  full_schema_rate    - fraction of outputs containing all 7 keys
  status_accuracy     - predicted status == gold status
  risk_p / risk_r / risk_f1 - micro-averaged; a predicted risk matches a gold risk when the
                        token-overlap F1 of their `risk` text is >= 0.5 (greedy one-to-one matching)
  risk_owner_acc      - owner correct among matched risks
  milestone_f1, next_steps_f1 - same overlap matching applied to milestone names / next steps
Usage: python src/evaluate.py --adapter outputs/.../adapter [--limit N]
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from prompting import prompt_text  # noqa: E402

KEYS = ["summary", "status", "milestones", "risks", "dependencies", "decisions_needed", "next_steps"]


def extract_json(text):
    t = re.sub(r"```(?:json)?", "", text).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e > s:
        try:
            obj = json.loads(t[s:e + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def toks(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def tok_f1(a, b):
    a, b = toks(a), toks(b)
    if not a or not b:
        return 0.0
    common = sum(min(a.count(w), b.count(w)) for w in set(a))
    if common == 0:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def match(pred, gold, thr=0.5):
    """Greedy one-to-one matching on token F1. Returns list of (pred_idx, gold_idx)."""
    pairs = sorted(((tok_f1(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold)), reverse=True)
    used_p, used_g, out = set(), set(), []
    for s, i, j in pairs:
        if s < thr:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); out.append((i, j))
    return out


def as_list(x):
    return x if isinstance(x, list) else []


def field_text(item, key):
    return item.get(key, "") if isinstance(item, dict) else item


def score(pred_text, gold):
    obj = extract_json(pred_text)
    try:
        strict = isinstance(json.loads(pred_text.strip()), dict)
    except Exception:
        strict = False
    r = {"valid": obj is not None, "strict": strict}
    obj = obj or {}
    present = [k in obj for k in KEYS]
    r["coverage"] = sum(present) / len(KEYS)
    r["full_schema"] = all(present)
    r["pred_status"] = str(obj.get("status", "none")).lower().strip()
    r["gold_status"] = gold["status"]
    if r["pred_status"] not in ("green", "yellow", "red"):
        r["pred_status"] = "other"
    r["status_ok"] = r["pred_status"] == gold["status"]
    pr = [field_text(x, "risk") for x in as_list(obj.get("risks"))]
    gr = [x["risk"] for x in gold["risks"]]
    m = match(pr, gr)
    r["risk_tp"], r["risk_np"], r["risk_ng"] = len(m), len(pr), len(gr)
    pred_risks = as_list(obj.get("risks"))
    r["owner_ok"] = sum(1 for i, j in m if isinstance(pred_risks[i], dict)
                        and str(pred_risks[i].get("owner", "")).lower() == gold["risks"][j]["owner"].lower())
    for name, key, sub in [("ms", "milestones", "name"), ("ns", "next_steps", None)]:
        p = [field_text(x, sub) if sub else x for x in as_list(obj.get(key))]
        g = [x[sub] if sub else x for x in gold[key]]
        r[f"{name}_tp"], r[f"{name}_np"], r[f"{name}_ng"] = len(match(p, g)), len(p), len(g)
    return r


def prf(tp, np_, ng):
    p = tp / np_ if np_ else 0.0
    rc = tp / ng if ng else 0.0
    return p, rc, (2 * p * rc / (p + rc) if p + rc else 0.0)


def aggregate(rows):
    n = len(rows)
    S = lambda k: sum(r[k] for r in rows)  # noqa: E731
    rp, rr, rf = prf(S("risk_tp"), S("risk_np"), S("risk_ng"))
    return {
        "n": n,
        "json_valid_rate": S("valid") / n, "strict_json_rate": S("strict") / n,
        "schema_coverage": S("coverage") / n, "full_schema_rate": S("full_schema") / n,
        "status_accuracy": S("status_ok") / n,
        "risk_precision": rp, "risk_recall": rr, "risk_f1": rf,
        "risk_owner_acc": S("owner_ok") / S("risk_tp") if S("risk_tp") else 0.0,
        "milestone_f1": prf(S("ms_tp"), S("ms_np"), S("ms_ng"))[2],
        "next_steps_f1": prf(S("ns_tp"), S("ns_np"), S("ns_ng"))[2],
        "status_confusion": {g: {p: sum(1 for r in rows if r["gold_status"] == g and r["pred_status"] == p)
                                 for p in ["green", "yellow", "red", "other"]} for g in ["green", "yellow", "red"]},
    }


@torch.no_grad()
def generate(model, tok, notes_list, max_new_tokens, bs):
    outs = []
    for i in range(0, len(notes_list), bs):
        prompts = [prompt_text(tok, n) for n in notes_list[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  generated {len(outs)}/{len(notes_list)}", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test_file", default="data/test.jsonl")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=448)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--rescore", action="store_true",
                    help="recompute metrics from an existing out_dir/predictions.jsonl without regenerating")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    rows = [json.loads(l) for l in open(a.test_file)]
    if a.limit:
        rows = rows[: a.limit]
    golds = [json.loads(r["target"]) for r in rows]
    os.makedirs(a.out_dir, exist_ok=True)
    results, preds = {}, {}
    if a.rescore:
        old = json.load(open(os.path.join(a.out_dir, "metrics.json")))
        prs = [json.loads(l) for l in open(os.path.join(a.out_dir, "predictions.jsonl"))]
        rows = [r for r in rows if r["id"] in {p["id"] for p in prs}]
        golds = [json.loads(r["target"]) for r in rows]
        for name in ["base", "finetuned"]:
            preds[name] = [p[name] for p in prs]
            scored = [score(p, g) for p, g in zip(preds[name], golds)]
            results[name] = {**aggregate(scored), "generation_seconds": old[name]["generation_seconds"]}
        return write_outputs(a, rows, golds, preds, results)
    tok = AutoTokenizer.from_pretrained(a.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.float32).eval()
    from peft import PeftModel
    for name in ["base", "finetuned"]:
        if name == "finetuned":
            model = PeftModel.from_pretrained(model, a.adapter).eval()
        print(f"[{name}] generating on {len(rows)} test examples", flush=True)
        t0 = time.time()
        preds[name] = generate(model, tok, [r["notes"] for r in rows], a.max_new_tokens, a.batch_size)
        secs = time.time() - t0
        scored = [score(p, g) for p, g in zip(preds[name], golds)]
        results[name] = {**aggregate(scored), "generation_seconds": round(secs, 1)}
        print(name, json.dumps(results[name], indent=2), flush=True)
    write_outputs(a, rows, golds, preds, results)


def write_outputs(a, rows, golds, preds, results):
    with open(os.path.join(a.out_dir, "predictions.jsonl"), "w") as f:
        for r, g, pb, pf in zip(rows, golds, preds["base"], preds["finetuned"]):
            f.write(json.dumps({"id": r["id"], "notes": r["notes"], "gold": g, "base": pb, "finetuned": pf}) + "\n")
    json.dump({"base_model": a.base_model, "adapter": a.adapter, "test_file": a.test_file, **results},
              open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=2)

    labels = [("JSON validity (lenient)", "json_valid_rate"), ("JSON validity (strict, raw output)", "strict_json_rate"),
              ("Schema-field coverage", "schema_coverage"), ("All 7 fields present", "full_schema_rate"),
              ("Status (RAG) accuracy", "status_accuracy"), ("Risk precision", "risk_precision"),
              ("Risk recall", "risk_recall"), ("Risk F1", "risk_f1"), ("Risk owner accuracy (matched)", "risk_owner_acc"),
              ("Milestone F1", "milestone_f1"), ("Next-steps F1", "next_steps_f1")]
    md = [f"Test set: `{a.test_file}` (n={len(rows)}), greedy decoding, max_new_tokens={a.max_new_tokens}\n",
          "| Metric | Base | Fine-tuned (LoRA) |", "|---|---|---|"]
    md += [f"| {lab} | {results['base'][k]:.3f} | {results['finetuned'][k]:.3f} |" for lab, k in labels]
    md.append(f"| Generation time (s) | {results['base']['generation_seconds']} | {results['finetuned']['generation_seconds']} |")
    md.append("\n**Status confusion (rows = gold, cols = predicted: green / yellow / red / other)**\n")
    md += ["| Model | Gold | green | yellow | red | other |", "|---|---|---|---|---|---|"]
    for name in ["base", "finetuned"]:
        for g, row in results[name]["status_confusion"].items():
            md.append(f"| {name} | {g} | " + " | ".join(str(row[p]) for p in ["green", "yellow", "red", "other"]) + " |")
    md.append("\n## Qualitative examples\n")
    for r, g, pb, pf in list(zip(rows, golds, preds["base"], preds["finetuned"]))[:3]:
        md += [f"### {r['id']}\n", "**Notes**\n", "```\n" + r["notes"] + "\n```", f"**Gold status:** {g['status']}\n",
               "**Base output**\n", "```\n" + pb.strip()[:1500] + "\n```", "**Fine-tuned output**\n",
               "```json\n" + pf.strip()[:2500] + "\n```\n"]
    open(os.path.join(a.out_dir, "results.md"), "w").write("\n".join(md))
    print("\n".join(md[:16]))


if __name__ == "__main__":
    main()
