"""Shared prompt construction so training, eval and demo use identical formatting."""

SYSTEM_PROMPT = (
    "You are a program management assistant. Convert the user's messy program or standup notes "
    "into a JSON program status report with exactly these keys: summary (string), status "
    "(green|yellow|red), milestones (list of {name, due, status: done|on_track|at_risk|delayed}), "
    "risks (list of {risk, impact: low|medium|high, owner, mitigation}), dependencies (list of strings), "
    "decisions_needed (list of strings), next_steps (list of strings). Status rule: red if any milestone "
    "is delayed or 2+ high-impact risks; yellow if any milestone at risk, 1 high-impact risk, or 3+ risks; "
    "otherwise green. Output only JSON."
)


def build_messages(notes, target=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": notes}]
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs


def prompt_text(tokenizer, notes):
    return tokenizer.apply_chat_template(build_messages(notes), tokenize=False, add_generation_prompt=True)
