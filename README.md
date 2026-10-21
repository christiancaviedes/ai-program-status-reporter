# Messy PM Notes → Structured Status Reports (LoRA fine-tuning of SmolLM2)

Fine-tunes small open models (**SmolLM2-135M-Instruct** and **SmolLM2-360M-Instruct**) with **LoRA (PEFT)** to turn
messy program-management notes into a **fixed JSON status report schema** (summary, RAG status, milestones, risks,
dependencies, decisions needed, next steps). The fine-tuned and base models are evaluated on the same held-out test set.
Everything here, training included, runs on a **CPU only** (no GPU).

> Built with AI-assisted development.

---

## The problem

Program and project managers spend a lot of time every week turning standup notes, meeting scribbles and chat
threads into status reports. The notes are inconsistent: shorthand ("tgt", "b4", "w/"), mixed bullets, off-topic
chatter, and risks described in many different ways. Leaders who read these reports need **consistent,
decision-ready data**:

- a clear **Red / Yellow / Green** status based on an explicit rule, not on gut feel
- milestones with dates and state
- risks with **impact, owner and mitigation**
- explicit **decisions needed** and **next steps**

A structured JSON output also feeds dashboards, portfolio roll-ups and RAID logs directly. This project tests
whether a *very small* model, cheap enough to run on a laptop CPU, can learn that transformation reliably.

## Target schema

```json
{
  "summary": "string",
  "status": "green | yellow | red",
  "milestones": [{"name": "...", "due": "...", "status": "done | on_track | at_risk | delayed"}],
  "risks": [{"risk": "...", "impact": "low | medium | high", "owner": "...", "mitigation": "..."}],
  "dependencies": ["..."],
  "decisions_needed": ["..."],
  "next_steps": ["..."]
}
```

**Status rule** (the system prompt states it and the data generator enforces it):
`red` if any milestone is delayed or there are 2+ high-impact risks; `yellow` if any milestone is at risk, there is
exactly 1 high-impact risk, or there are 3+ risks; otherwise `green`.

## Dataset (synthetic)

**All data is synthetic.** No real company or project notes were used. `src/generate_dataset.py` builds the dataset
from templates with a fixed seed (`--n 480 --seed 42`):

1. **Sample a ground-truth program state:** a project (20 fictional names), 2–4 milestones (16 types) with due
   dates and states, 0–3 risks drawn from a 12-item risk catalogue (each has several paraphrases and mitigations),
   owners (20 first names), dependencies, decisions needed and next steps. The target RAG class is sampled first to
   balance the classes. The final label is then always computed with the deterministic status rule above.
2. **Render messy notes** from that state: varied phrasings ("slipped past", "tight for … prob slips"), impact
   phrased informally ("could kill the date", "not a huge deal"), random lowercasing, abbreviations
   (w/, b/c, b4, mtg, env, sched), mixed bullet styles, shuffled line order, and irrelevant noise lines
   ("coffee machine on 3rd floor broken again lol").
3. **Render the gold JSON** directly from the state. A risk's `risk` field is a *canonical* label (e.g. notes saying
   "vendor still hasn't shipped the api" map to `"vendor API delivery delayed"`), so the model has to normalise the
   text, not copy it.

| Split | Examples | green | yellow | red |
|---|---|---|---|---|
| train | 384 | 117 | 110 | 157 |
| validation | 48 | 20 | 19 | 9 |
| test | 48 | 11 | 15 | 22 |
| **total** | **480** | | | |

(Counts from `data/stats.json`.) The splits are in `data/*.jsonl` and as a Hugging Face `DatasetDict` in
`data/hf_dataset/`.

## Method

| | |
|---|---|
| Base models | `HuggingFaceTB/SmolLM2-135M-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct` |
| Approach | Supervised fine-tuning with LoRA (`peft`) via `transformers.Trainer`; chat template with a system prompt (`src/prompting.py`) |
| Loss masking | Loss computed on assistant (JSON) tokens only; prompt tokens masked with -100 |
| LoRA | r = 16, alpha = 32, dropout = 0.05, target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable params | 135M: 4,884,480 of 139,399,488 (3.5039%) · 360M: 8,683,520 of 370,504,640 (2.3437%) |
| Epochs / steps | 2 epochs, 96 optimizer steps (both models) |
| Effective batch | 8 (135M: 4 × grad-accum 2 · 360M: 2 × grad-accum 4) |
| Optimiser | AdamW (Trainer default), LR 3e-4, cosine schedule, 5% warmup, weight decay 0 |
| Max sequence length | 768 tokens |
| Precision | float32 |
| Seed | 42 |
| Hardware | CPU only: Intel Xeon, 8 threads, ~15 GB RAM, no GPU |
| Software | Python 3.13.5, torch 2.14.1+cpu, transformers 5.20.2, peft 0.21.2, datasets 5.1.0, accelerate 1.15.0 |

Configs: `configs/smollm2_135m_lora.yaml`, `configs/smollm2_360m_lora.yaml`.

### Training runs

| Model | Train time (s) | Final train loss (avg) | Final validation loss |
|---|---|---|---|
| SmolLM2-135M + LoRA | 1161.7648 | 0.4110434638957183 | 0.19709819555282593 |
| SmolLM2-360M + LoRA | 1960.5068 | 0.252467275209104 | 0.11335456371307373 |

(From `outputs/*/train_log.jsonl` and `outputs/*/train_summary.json`. That is about 19 and 33 minutes of CPU training.)

## Evaluation

`src/evaluate.py` runs the **base model** and the **base model + LoRA adapter** on the 48 held-out test examples,
with greedy decoding, `max_new_tokens=448` and the same prompt for both.

Metrics:
- **JSON validity:** lenient (after stripping code fences / surrounding text) and strict (raw output parses)
- **Schema-field coverage / All 7 fields present:** required top-level keys present
- **Status (RAG) accuracy:** exact match on green / yellow / red
- **Risk P / R / F1:** micro-averaged; a predicted risk matches a gold risk if token-overlap F1 of the `risk` text is ≥ 0.5 (greedy one-to-one matching)
- **Risk owner accuracy:** correct owner among matched risks
- **Milestone F1, Next-steps F1:** the same overlap matching on milestone names / next-step text

### Results: SmolLM2-135M-Instruct (`results/smollm2-135m/`)

Test set: `data/test.jsonl` (n=48), greedy decoding, max_new_tokens=448

| Metric | Base | Fine-tuned (LoRA) |
|---|---|---|
| JSON validity (lenient) | 0.000 | 0.958 |
| JSON validity (strict, raw output) | 0.000 | 0.958 |
| Schema-field coverage | 0.000 | 0.958 |
| All 7 fields present | 0.000 | 0.958 |
| Status (RAG) accuracy | 0.000 | 0.417 |
| Risk precision | 0.000 | 0.307 |
| Risk recall | 0.000 | 0.342 |
| Risk F1 | 0.000 | 0.323 |
| Risk owner accuracy (matched) | 0.000 | 0.519 |
| Milestone F1 | 0.000 | 0.738 |
| Next-steps F1 | 0.000 | 0.729 |
| Generation time (s) | 138.2 | 107.6 |

### Results: SmolLM2-360M-Instruct (`results/smollm2-360m/`)

Test set: `data/test.jsonl` (n=48), greedy decoding, max_new_tokens=448

| Metric | Base | Fine-tuned (LoRA) |
|---|---|---|
| JSON validity (lenient) | 0.417 | 0.979 |
| JSON validity (strict, raw output) | 0.417 | 0.979 |
| Schema-field coverage | 0.417 | 0.979 |
| All 7 fields present | 0.417 | 0.979 |
| Status (RAG) accuracy | 0.083 | 0.625 |
| Risk precision | 0.000 | 0.600 |
| Risk recall | 0.000 | 0.570 |
| Risk F1 | 0.000 | 0.584 |
| Risk owner accuracy (matched) | 0.000 | 0.978 |
| Milestone F1 | 0.173 | 0.789 |
| Next-steps F1 | 0.205 | 0.886 |
| Generation time (s) | 232.8 | 214.5 |

### Status confusion matrices (rows = gold, columns = predicted)

| Model | Gold | green | yellow | red | other |
|---|---|---|---|---|---|
| 135M fine-tuned | green | 2 | 1 | 7 | 1 |
| 135M fine-tuned | yellow | 5 | 1 | 9 | 0 |
| 135M fine-tuned | red | 2 | 2 | 17 | 1 |
| 360M fine-tuned | green | 10 | 0 | 1 | 0 |
| 360M fine-tuned | yellow | 5 | 2 | 8 | 0 |
| 360M fine-tuned | red | 2 | 1 | 18 | 1 |

Base-model matrices and per-example predictions are in `results/<model>/results.md` and `predictions.jsonl`.

### Takeaways

- **Format is learned quickly.** With 384 training examples and about 20–33 CPU-minutes of LoRA training, both models
  went from mostly unusable output (the base 135M never produced parseable JSON; the base 360M did in 0.417 of
  cases) to valid, complete JSON in 0.958 (135M) and 0.979 (360M) of test cases.
- **Extraction is better than judgement.** Milestones and next steps, which are mostly copy-and-normalise tasks, score
  much higher than the RAG status, which needs reasoning over several fields.
- **Model size matters for reasoning.** The 360M adapter beats the 135M adapter on every metric. The 135M
  model's status accuracy (0.417) is below a trivial "always red" baseline on this test set (22/48 = 0.458);
  the 360M model (0.625) beats that baseline.
- **"Yellow" is the hard class.** Both fine-tuned models mostly misclassify yellow (2/15 correct for 360M), usually
  as red. Yellow needs the most careful use of the rule (one high risk vs. two, at-risk vs. delayed).

## Example (test example `syn-0156`, SmolLM2-360M)

**Input notes**

```
Zephyr weekly sync
> cutover rehearsal tight for Oct 23, prob slips a few days
- jira board needs cleanup
-- depends on security to deliver SDK update
• Sam joined late
go/no-go mtg slipped past Mar 23
• action: Fatima to update RAID log
- AI: Kofi to set up sync with backend
> license expires soon, not renewed (could kill the date) owner: hannah -> escalate renewal to legal lead
```

**Fine-tuned output (SmolLM2-360M + LoRA)**

```json
{"summary": "Zephyr cloud migration is off track; 0/2 milestones complete, license renewal need attention, 1 open risk(s).", "status": "red", "milestones": [{"name": "cutover rehearsal", "due": "Oct 23", "status": "at_risk"}, {"name": "go/no-go meeting", "due": "Mar 23", "status": "delayed"}], "risks": [{"risk": "third-party license renewal pending", "impact": "high", "owner": "Hannah", "mitigation": "escalate renewal to legal lead"}], "dependencies": ["security to deliver SDK update"], "decisions_needed": [], "next_steps": ["Fatima to update RAID log", "Kofi to set up sync with backend"]}
```

Status, milestones, the risk (normalised label, impact, owner, mitigation), the dependency and the next steps all
match the gold report, and the noise lines are ignored. Two errors in the summary sentence: the notes only say
"Zephyr", and the model **made up** the project name "cloud migration" (gold: "Zephyr API gateway upgrade"). It
also listed "license renewal" where the gold lists the at-risk milestones. Free-text fields need human review.

**Base model output (SmolLM2-360M, no adapter), same input:** valid JSON but wrong status (`green`), a made-up risk
owner, and lines copied into the wrong fields:

```json
{"summary": "Zephyr weekly sync", "status": "green",
 "milestones": [{"name": "cutover rehearsal tight for Oct 23", "due": "Oct 23", "status": "at_risk"}],
 "risks": [{"impact": "high", "owner": "Sam", "mitigation": "no"}],
 "dependencies": ["go/no-go mtg slipped past Mar 23", "AI: Kofi to set up sync with backend"],
 "decisions_needed": ["Fatima to update RAID log"], "next_steps": ["Fatima to update RAID log"]}
```
(The base output is reformatted onto fewer lines here; the content is unchanged.)

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Full pipeline (data -> train -> eval) for one model:
./run.sh configs/smollm2_135m_lora.yaml     # ~20 min training on 8 CPU threads
./run.sh configs/smollm2_360m_lora.yaml     # ~33 min training on 8 CPU threads

# Or step by step:
make data
make train CONFIG=configs/smollm2_360m_lora.yaml
make eval  BASE=HuggingFaceTB/SmolLM2-360M-Instruct \
           ADAPTER=outputs/smollm2-360m-pm-status-lora/adapter RESULTS=results/smollm2-360m

# Try it on your own notes:
python src/demo.py --base_model HuggingFaceTB/SmolLM2-135M-Instruct \
  --adapter outputs/smollm2-135m-pm-status-lora/adapter --notes_file my_notes.txt
# add --gradio for a small web UI (pip install gradio)
```

To recompute metrics from saved predictions without generating again:
`python src/evaluate.py --base_model <base> --adapter <adapter> --out_dir results/<model> --rescore`.

**Artifacts in the repo:** the SmolLM2-135M LoRA adapter (`outputs/smollm2-135m-pm-status-lora/adapter/`,
about 19.6 MB) is committed. The **SmolLM2-360M adapter (about 34.8 MB) is not committed** because of its size.
Regenerate it with `python src/train.py --config configs/smollm2_360m_lora.yaml`. The training logs and summary for
both runs are committed. Tokenizer files are loaded from the base model on the Hub.

## Project layout

```
configs/            LoRA training configs (135M, 360M)
data/               synthetic train/validation/test splits (+ HF DatasetDict, stats.json)
src/generate_dataset.py   seeded synthetic data generator
src/prompting.py    shared system prompt + chat formatting (train/eval/demo use the same one)
src/train.py        LoRA SFT with transformers.Trainer + peft
src/evaluate.py     base vs fine-tuned evaluation and metrics
src/demo.py         CLI / Gradio demo
outputs/            adapters, training logs and summaries
results/<model>/    metrics.json, results.md, predictions.jsonl
run.sh, Makefile    end-to-end reproduction
```

## Limitations

- **Synthetic, template-based data.** Train and test examples come from the same generator, templates and
  vocabulary (20 projects, 16 milestone types, 12 risk types), so the test set is **in-distribution**. These scores
  are very likely **higher than real-world performance** on real notes, which have different vocabulary, longer
  context, ambiguity and implicit information.
- **Small test set** (n = 48, 11 / 15 / 22 per class). One example is about 2 percentage points, so differences
  of a few points are not meaningful. Each model was trained once (seed 42), with no variance estimate.
- **Lexical metrics.** Risk, milestone and next-step matching uses token-overlap F1 ≥ 0.5. That can credit
  near-misses and miss correct paraphrases. There is no human or LLM-judge evaluation.
- **Free text can be made up.** The models sometimes invent project names, assign the wrong risk label, or duplicate
  list items (see `results/*/results.md`). The output is a draft for a PM to review, not a source of truth.
- **The RAG status is a fixed toy rule.** Real organisations use different, often subjective criteria. Because the
  rule is deterministic, a post-processing step that computes status from the extracted fields would likely beat
  the model's own status prediction.
- **Tiny models, CPU only, short training** (2 epochs, no hyperparameter search). Larger models or more and more
  varied data would probably help, especially with the "yellow" class.

## License

MIT, see `LICENSE`.
