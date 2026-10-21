"""Seeded synthetic data generator: messy program/standup notes -> structured JSON status report.

ALL DATA PRODUCED BY THIS SCRIPT IS SYNTHETIC. Projects, people, dates and events are
randomly composed from templates below; any resemblance to real programs is coincidental.

Design:
  1. Sample a latent "program state" (project, milestones with states, risks, deps,
     decisions, next steps, overall health).
  2. Render that state as messy notes: shuffled lines, abbreviations, lowercase,
     filler/noise lines, inconsistent bullets, several phrasings per fact.
  3. Render the same state as the canonical JSON target.
Overall status is a deterministic function of the latent state (see `derive_status`),
so the label is recoverable from the notes, not memorised noise.

Usage: python src/generate_dataset.py --n 480 --seed 42 --out data
"""
import argparse
import json
import os
import random

PROJECTS = [
    "Atlas data platform migration", "Phoenix mobile app relaunch", "Orion billing revamp",
    "Helios SSO rollout", "Nimbus cloud cost reduction", "Vega payments integration",
    "Lighthouse analytics dashboard", "Mercury CRM consolidation", "Titan warehouse automation",
    "Aurora customer portal", "Zephyr API gateway upgrade", "Pegasus ML fraud model",
    "Polaris HR system cutover", "Cobalt network refresh", "Juniper search relevance",
    "Sierra SOC2 readiness", "Falcon checkout redesign", "Granite ERP upgrade",
    "Echo support chatbot", "Kestrel observability stack",
]
PEOPLE = ["Priya", "Marcus", "Elena", "Tom", "Aisha", "Diego", "Sam", "Wei", "Fatima", "Jonas",
          "Grace", "Raj", "Lucia", "Ben", "Hannah", "Omar", "Mei", "Carlos", "Nina", "Kofi"]
TEAMS = ["platform", "infra", "security", "data eng", "mobile", "frontend", "backend", "QA",
         "legal", "finance", "vendor mgmt", "SRE", "design", "procurement"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MILESTONES = ["design review", "beta launch", "UAT sign-off", "data migration", "load testing",
              "security review", "GA release", "pilot with 3 customers", "vendor contract signed",
              "API freeze", "code complete", "training rollout", "cutover rehearsal",
              "pen test", "go/no-go meeting", "MVP demo"]
# risk: (canonical text, messy phrasings, default impact, mitigation options)
RISKS = [
    ("vendor API delivery delayed", ["vendor still hasn't shipped the api", "vendor api late again",
     "waiting on vendor API, eta unclear"], ["follow up with vendor account manager", "build a mock API to unblock testing"]),
    ("key engineer on leave", ["{p2} out on leave for 3 wks", "losing {p2} to PTO next sprint",
     "{p2} OOO, no backup yet"], ["cross-train a backup engineer", "re-sequence work to cover the gap"]),
    ("budget overrun on cloud spend", ["cloud bill way over forecast", "aws spend trending ~20% over",
     "burning budget faster than planned on cloud"], ["request budget review with finance", "right-size instances"]),
    ("security review findings unresolved", ["sec review came back w/ open findings",
     "still have unresolved security findings", "security flagged issues not fixed yet"], ["schedule fix sprint for findings", "get risk acceptance from CISO"]),
    ("scope creep from stakeholder requests", ["stakeholders keep adding asks", "new feature requests creeping in",
     "scope keeps growing from sales asks"], ["enforce change control process", "push new asks to phase 2"]),
    ("data quality issues in legacy system", ["legacy data is a mess, lots of dupes", "found bad data in legacy tables",
     "data quality in old system worse than expected"], ["run data cleansing job before migration", "add validation checks"]),
    ("test environment instability", ["test env keeps falling over", "staging flaky all week",
     "QA env unstable, tests failing randomly"], ["dedicate SRE support to stabilize env", "spin up a second test env"]),
    ("hiring delays for contractors", ["contractors still not onboarded", "hiring for contractors slipping",
     "no contractor start date yet"], ["escalate to procurement", "borrow staff from another team"]),
    ("third-party license renewal pending", ["license renewal still pending w/ legal", "license expires soon, not renewed",
     "waiting on legal for license renewal"], ["escalate renewal to legal lead", "secure temporary extension"]),
    ("performance below SLA in load tests", ["load tests show latency over SLA", "perf numbers not meeting SLA",
     "p99 latency too high in load test"], ["profile and optimize hot paths", "add caching layer"]),
    ("integration dependency on another team's release", ["{team} release we depend on might slip", "blocked until {team} ships their piece",
     "need {team} to deliver first"], ["agree delivery date with {team} lead", "decouple via feature flag"]),
    ("regulatory approval timeline uncertain", ["compliance approval timing unclear", "regulator sign-off date unknown",
     "still no word from compliance on approval"], ["engage compliance early with draft docs", "plan launch without regulated features"]),
]
DEPENDENCIES = ["{team} team to provision environments", "{team} sign-off on architecture",
                "{team} to deliver SDK update", "{team} approval of data retention policy",
                "{team} to complete network changes", "{team} review of contract terms",
                "{team} to finish identity provider config"]
DECISIONS = ["approve additional budget for {x}", "choose between build vs buy for {x}",
             "decide whether to delay launch by two weeks", "agree on cutover date",
             "prioritize {x} vs reporting features", "approve hiring two contractors",
             "confirm go-live scope for phase 1", "pick vendor for {x}"]
DECISION_X = ["monitoring", "data pipeline", "the auth layer", "load testing", "the mobile SDK", "support tooling"]
NEXT_STEPS = ["{p} to send updated timeline by {d}", "{p} to schedule risk review with sponsors",
              "{p} to finalize test plan", "{p} to update RAID log", "{p} to set up sync with {team}",
              "{p} to draft go/no-go criteria", "{p} to circulate status deck", "{p} to book UAT sessions",
              "{p} to confirm resourcing with {team} lead"]
NOISE = ["coffee machine on 3rd floor broken again lol", "reminder: offsite next thurs",
         "someone pls fix the conf room booking", "(notes taken quickly, sorry for typos)",
         "jira board needs cleanup", "{p} joined late", "recording link in chat",
         "happy bday {p}!", "TODO: rename the channel", "audio issues at start"]
MS_STATE_PHRASES = {
    "done": ["{m} done ✅ ({d})", "finished {m} on {d}", "{m} - complete as of {d}", "wrapped {m} ({d})"],
    "on_track": ["{m} on track for {d}", "{m} still looking good for {d}", "{m} tgt {d}, on schedule"],
    "at_risk": ["{m} at risk for {d}", "{m} ({d}) might slip", "{m} tight for {d}, prob slips a few days"],
    "delayed": ["{m} slipped past {d}", "{m} missed {d}, now late", "{m} delayed - was due {d}"],
}
IMPACT_PHRASES = {"high": ["big deal", "critical", "sev1-ish", "could kill the date"],
                  "medium": ["moderate impact", "medium", "annoying but manageable"],
                  "low": ["minor", "low impact", "not a huge deal"]}
SUMMARY_LEAD = {"green": "{proj} is on track", "yellow": "{proj} is at risk",
                "red": "{proj} is off track"}


def date(rng):
    return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}"


def fill(s, rng, **kw):
    return s.format(p=kw.get("p", rng.choice(PEOPLE)), p2=kw.get("p2", rng.choice(PEOPLE)),
                    team=kw.get("team", rng.choice(TEAMS)), d=kw.get("d", date(rng)),
                    x=kw.get("x", rng.choice(DECISION_X)), m=kw.get("m", ""))


def derive_status(milestones, risks):
    """Deterministic RAG rule (documented in README)."""
    n_delayed = sum(m["status"] == "delayed" for m in milestones)
    n_at_risk = sum(m["status"] == "at_risk" for m in milestones)
    n_high = sum(r["impact"] == "high" for r in risks)
    if n_delayed >= 1 or n_high >= 2:
        return "red"
    if n_at_risk >= 1 or n_high == 1 or len(risks) >= 3:
        return "yellow"
    return "green"


def sample_state(rng):
    proj = rng.choice(PROJECTS)
    target = rng.choice(["green", "yellow", "red"])  # balance classes, then enforce via rule
    ms_names = rng.sample(MILESTONES, rng.randint(2, 4))
    if target == "green":
        states = [rng.choice(["done", "on_track"]) for _ in ms_names]
        n_risks, impacts = rng.randint(0, 2), ["low", "medium"]
    elif target == "yellow":
        states = [rng.choice(["done", "on_track", "at_risk"]) for _ in ms_names]
        n_risks, impacts = rng.randint(1, 3), ["low", "medium", "high"]
    else:
        states = [rng.choice(["done", "on_track", "at_risk", "delayed"]) for _ in ms_names]
        n_risks, impacts = rng.randint(1, 3), ["medium", "high", "high"]
    milestones = [{"name": n, "due": date(rng), "status": s} for n, s in zip(ms_names, states)]
    risks = []
    for spec in rng.sample(RISKS, n_risks):
        canon, phrasings, mitigations = spec
        team, p2 = rng.choice(TEAMS), rng.choice(PEOPLE)
        impact = rng.choice(impacts)
        if target == "green":
            impact = rng.choice(["low", "medium"])
        risks.append({
            "risk": canon, "impact": impact, "owner": rng.choice(PEOPLE),
            "mitigation": fill(rng.choice(mitigations), rng, team=team),
            "_phrase": fill(rng.choice(phrasings), rng, team=team, p2=p2),
        })
    if target == "red" and not any(m["status"] == "delayed" for m in milestones) \
            and sum(r["impact"] == "high" for r in risks) < 2:
        milestones[rng.randrange(len(milestones))]["status"] = "delayed"
    if target == "yellow" and derive_status(milestones, risks) == "green":
        milestones[rng.randrange(len(milestones))]["status"] = "at_risk"
    status = derive_status(milestones, risks)
    deps = [fill(d, rng) for d in rng.sample(DEPENDENCIES, rng.randint(0, 2))]
    decs = [fill(d, rng) for d in rng.sample(DECISIONS, rng.randint(0, 2))]
    steps = [fill(s, rng) for s in rng.sample(NEXT_STEPS, rng.randint(1, 3))]
    return dict(project=proj, milestones=milestones, risks=risks, dependencies=deps,
                decisions_needed=decs, next_steps=steps, status=status)


def messify(line, rng):
    r = rng.random()
    if r < 0.3:
        line = line.lower()
    for a, b in [("with", "w/"), ("because", "b/c"), ("before", "b4"), ("meeting", "mtg"),
                 ("environment", "env"), ("schedule", "sched")]:
        if rng.random() < 0.4:
            line = line.replace(a, b)
    bullet = rng.choice(["- ", "* ", "• ", "", "-- ", "> "])
    return bullet + line


def render_notes(st, rng):
    lines = []
    for m in st["milestones"]:
        lines.append(fill(rng.choice(MS_STATE_PHRASES[m["status"]]), rng, m=m["name"], d=m["due"]))
    for r in st["risks"]:
        bits = [r["_phrase"], f"({rng.choice(IMPACT_PHRASES[r['impact']])})",
                rng.choice([f"{r['owner']} owns", f"owner: {r['owner']}", f"{r['owner']} on it"]),
                rng.choice([f"plan: {r['mitigation']}", f"-> {r['mitigation']}", f"mitigation = {r['mitigation']}"])]
        lines.append(" ".join(bits))
    for d in st["dependencies"]:
        lines.append(rng.choice(["dep: ", "depends on ", "need "]) + d)
    for d in st["decisions_needed"]:
        lines.append(rng.choice(["DECISION NEEDED: ", "need decision - ", "sponsors to decide: ", "?? "]) + d)
    for s in st["next_steps"]:
        lines.append(rng.choice(["AI: ", "action: ", "next: ", ""]) + s)
    for _ in range(rng.randint(0, 2)):
        lines.append(fill(rng.choice(NOISE), rng))
    rng.shuffle(lines)
    header = rng.choice([f"{st['project']} standup {date(rng)}", f"notes - {st['project'].lower()}",
                         f"{st['project'].split()[0]} weekly sync", f"mtg notes {date(rng)} ({st['project']})"])
    return header + "\n" + "\n".join(messify(l, rng) for l in lines)


def render_target(st):
    n_done = sum(m["status"] == "done" for m in st["milestones"])
    bad = [m["name"] for m in st["milestones"] if m["status"] in ("at_risk", "delayed")]
    summary = SUMMARY_LEAD[st["status"]].format(proj=st["project"]) + \
        f"; {n_done}/{len(st['milestones'])} milestones complete"
    if bad:
        summary += f", {', '.join(bad)} need attention"
    summary += f", {len(st['risks'])} open risk(s)."
    return {
        "summary": summary,
        "status": st["status"],
        "milestones": st["milestones"],
        "risks": [{k: r[k] for k in ("risk", "impact", "owner", "mitigation")} for r in st["risks"]],
        "dependencies": st["dependencies"],
        "decisions_needed": st["decisions_needed"],
        "next_steps": st["next_steps"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for i in range(a.n):
        st = sample_state(rng)
        rows.append({"id": f"syn-{i:04d}", "notes": render_notes(st, rng),
                     "target": json.dumps(render_target(st), ensure_ascii=False), "synthetic": True})
    rng.shuffle(rows)
    n_test, n_val = int(a.n * a.test_frac), int(a.n * a.val_frac)
    splits = {"test": rows[:n_test], "validation": rows[n_test:n_test + n_val], "train": rows[n_test + n_val:]}
    os.makedirs(a.out, exist_ok=True)
    for name, rs in splits.items():
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    try:
        from datasets import Dataset, DatasetDict
        DatasetDict({k: Dataset.from_list(v) for k, v in splits.items()}).save_to_disk(os.path.join(a.out, "hf_dataset"))
    except ImportError:
        print("datasets not installed; skipped save_to_disk")
    stats = {k: len(v) for k, v in splits.items()}
    for k, v in splits.items():
        c = {}
        for r in v:
            s = json.loads(r["target"])["status"]
            c[s] = c.get(s, 0) + 1
        stats[f"{k}_status"] = c
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
