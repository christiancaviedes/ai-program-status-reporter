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

**Status confusion (rows = gold, cols = predicted: green / yellow / red / other)**

| Model | Gold | green | yellow | red | other |
|---|---|---|---|---|---|
| base | green | 4 | 0 | 0 | 7 |
| base | yellow | 6 | 0 | 0 | 9 |
| base | red | 10 | 0 | 0 | 12 |
| finetuned | green | 10 | 0 | 1 | 0 |
| finetuned | yellow | 5 | 2 | 8 | 0 |
| finetuned | red | 2 | 1 | 18 | 1 |

## Qualitative examples

### syn-0163

**Notes**

```
mtg notes Nov 7 (Pegasus ML fraud model)
-- security flagged issues not fixed yet (not a huge deal) jonas owns mitigation = sched fix sprint for findings
-- ga release - complete as of mar 9
• happy bday Diego!
- need security team to provision environments
> vendor still hasn't shipped the api (could kill the date) owner: Sam -> build a mock API to unblock testing
> cutover rehearsal done ✅ (Nov 21)
• need legal to finish identity provider config
> test env keeps falling over (not a huge deal) Ben owns -> dedicate SRE support to stabilize env
• action: lucia to set up sync w/ backend
```
**Gold status:** yellow

**Base output**

```
{
  "summary": "Pegasus ML fraud model",
  "status": "green",
  "milestones": [
    {
      "name": "Schedule Fixes",
      "due": "Mar 9",
      "status": "done"
    },
    {
      "name": "API Provisioning",
      "due": "Mar 9",
      "status": "done"
    },
    {
      "name": "Legal Review",
      "due": "Mar 9",
      "status": "done"
    },
    {
      "name": "Identity Provider Config",
      "due": "Mar 9",
      "status": "done"
    },
    {
      "name": "Test Environment",
      "due": "Mar 9",
      "status": "done"
    }
  ],
  "risks": [
    {
      "impact": "low",
      "owner": "Sam",
      "mitigation": "build a mock API to unblock testing"
    },
    {
      "impact": "medium",
      "owner": "Ben",
      "mitigation": "dedicate SRE support to stabilize env"
    },
    {
      "impact": "high",
      "owner": "Sam",
      "mitigation": "need legal to finish identity provider config"
    }
  ],
  "dependencies": [
    "Pegasus ML fraud model",
    "Pegasus ML fraud model"
  ],
  "decisions_needed": [
    "Pegasus ML fraud model",
    "Pegasus ML fraud model"
  ],
  "next_steps": [
    "Pegasus ML fraud model",
    "Pegasus ML fraud model"
  ]
}
```
**Fine-tuned output**

```json
{"summary": "Pegasus ML fraud model is on track; 0/3 milestones complete, 1 open risk(s).", "status": "green", "milestones": [{"name": "cutover rehearsal", "due": "Nov 21", "status": "done"}, {"name": "security review", "due": "Mar 9", "status": "done"}, {"name": "test environment", "due": "Mar 9", "status": "at_risk"}], "risks": [{"risk": "vendor API delivery delayed", "impact": "high", "owner": "Sam", "mitigation": "build a mock API to unblock testing"}, {"risk": "security review findings unresolved", "impact": "low", "owner": "Jonas", "mitigation": "schedule fix sprint for findings"}], "dependencies": ["security team to provision environments", "legal to finish identity provider config"], "decisions_needed": [], "next_steps": ["Lucia to set up sync with backend", "Lucia to set up sync with backend"]}
```

### syn-0432

**Notes**

```
mtg notes Jan 8 (Falcon checkout redesign)
-- mvp demo tight for feb 23, prob slips a few days
* perf numbers not meeting SLA (could kill the date) Carlos on it -> profile and optimize hot paths
test env keeps falling over (critical) owner: jonas -> spin up a second test env
> next: Mei to send updated timeline by Apr 27
> training rollout slipped past Nov 13
> reminder: offsite next thurs
dep: procurement sign-off on architecture
- data quality in old system worse than expected (moderate impact) Fatima owns mitigation = run data cleansing job b4 migration
* dep: legal to deliver SDK update
- action: Mei to book UAT sessions
```
**Gold status:** red

**Base output**

```
{
  "summary": "mtg notes Jan 8 (Falcon checkout redesign)",
  "status": "green",
  "milestones": [
    {
      "name": "mtg notes Jan 8 (Falcon checkout redesign)",
      "due": "Feb 23",
      "status": "at_risk"
    }
  ],
  "risks": [
    {
      "impact": "medium",
      "owner": "jonas",
      "mitigation": "run data cleansing job b4 migration"
    }
  ],
  "dependencies": [
    "procurement sign-off on architecture",
    "data quality in old system worse than expected"
  ],
  "decisions_needed": [
    " Mei to book UAT sessions"
  ],
  "next_steps": [
    " Mei to book UAT sessions"
  ]
}
```
**Fine-tuned output**

```json
{"summary": "Falcon checkout redesign is off track; 0/4 milestones complete, MVP demo, training rollout, data quality need attention, 2 open risk(s).", "status": "red", "milestones": [{"name": "MVP demo", "due": "Feb 23", "status": "at_risk"}, {"name": "training rollout", "due": "Nov 13", "status": "delayed"}, {"name": "data quality in old system", "due": "May 10", "status": "high"}, {"name": "data quality in new system", "due": "May 10", "status": "high"}], "risks": [{"risk": "performance below SLA in test environment", "impact": "medium", "owner": "Carlos", "mitigation": "profile and optimize hot paths"}, {"risk": "integration dependency on another team's release", "impact": "high", "owner": "Jonas", "mitigation": "spin up a second test environment"}, {"risk": "key engineer on leave", "impact": "high", "owner": "Fatima", "mitigation": "run data cleansing job before migration"}], "dependencies": ["procurement sign-off on architecture", "data quality in old system"], "decisions_needed": [], "next_steps": ["Mei to send updated timeline by Apr 27"]}
```

### syn-0049

**Notes**

```
Mercury CRM consolidation standup Dec 21
- perf numbers not meeting SLA (could kill the date) Wei owns mitigation = profile and optimize hot paths
* API freeze done ✅ (Dec 19)
> next: Aisha to update RAID log
need frontend sign-off on architecture
-- next: omar to book uat sessions
finished beta launch on apr 24
-- losing Aisha to PTO next sprint (medium) owner: Lucia -> cross-train a backup engineer
-- coffee machine on 3rd floor broken again lol
- (notes taken quickly, sorry for typos)
```
**Gold status:** yellow

**Base output**

```
{
  "summary": "Mercury CRM Consolidation Standup",
  "status": "green",
  "milestones": [
    {
      "name": "API freeze done",
      "due": "Dec 19",
      "status": "done"
    },
    {
      "name": "Aisha to update RAID log",
      "due": "Dec 19",
      "status": "done"
    },
    {
      "name": "Omar to book uat sessions",
      "due": "Dec 21",
      "status": "done"
    },
    {
      "name": "Finished beta launch on apr 24",
      "due": "Dec 21",
      "status": "done"
    },
    {
      "name": "Losing Aisha to PTO next sprint",
      "due": "Dec 21",
      "status": "yellow"
    },
    {
      "name": "Coffee machine on 3rd floor broken again",
      "due": "Dec 21",
      "status": "yellow"
    },
    {
      "name": "Need frontend sign-off on architecture",
      "due": "Dec 21",
      "status": "yellow"
    },
    {
      "name": "Next steps: omar to book uat sessions",
      "due": "Dec 21",
      "status": "yellow"
    }
  ],
  "risks": [
    {
      "impact": "low",
      "owner": "Lucia",
      "mitigation": "optimize hot paths"
    },
    {
      "impact": "medium",
      "owner": "Lucia",
      "mitigation": "optimize hot paths"
    },
    {
      "impact": "high",
      "owner": "Lucia",
      "mitigation": "optimize hot paths"
    },
    {
      "impact": "medium",
      "owner": "Lucia",
      "mitigation": "optimize hot paths"
    },
    {
      "impact": "medium",
      "owner": "Lucia",
```
**Fine-tuned output**

```json
{"summary": "Mercury CRM consolidation is off track; 1/2 milestones complete, 2 open risk(s).", "status": "red", "milestones": [{"name": "API freeze", "due": "Dec 19", "status": "done"}, {"name": "beta launch", "due": "Apr 24", "status": "done"}], "risks": [{"risk": "performance below SLA in load tests", "impact": "high", "owner": "Wei", "mitigation": "profile and optimize hot paths"}, {"risk": "key engineer on leave", "impact": "high", "owner": "Lucia", "mitigation": "cross-train a backup engineer"}], "dependencies": [], "decisions_needed": [], "next_steps": ["Aisha to update RAID log", "Omar to book UAT sessions"]}
```
