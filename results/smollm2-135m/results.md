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

**Status confusion (rows = gold, cols = predicted: green / yellow / red / other)**

| Model | Gold | green | yellow | red | other |
|---|---|---|---|---|---|
| base | green | 0 | 0 | 0 | 11 |
| base | yellow | 0 | 0 | 0 | 15 |
| base | red | 0 | 0 | 0 | 22 |
| finetuned | green | 2 | 1 | 7 | 1 |
| finetuned | yellow | 5 | 1 | 9 | 0 |
| finetuned | red | 2 | 2 | 17 | 1 |

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
You can convert the user's messy program or standup notes into a JSON program status report with exactly these keys: summary (string), status (green|yellow|red), milestones (list of {name, due, status: done|on_track|at_risk|delayed}), risks (list of {risk, impact: low|medium|high, owner, mitigation}), dependencies (list of strings), decisions_needed (list of strings), next_steps (list of strings). Status rule: red if any milestone is delayed or 2+ high-impact risks; yellow if any milestone at risk, 1 high-impact risk, or 3+ risks; otherwise green. Output only JSON.
```
**Fine-tuned output**

```json
{"summary": "Pegasus ML fraud model is off track; 1/4 milestones complete, GA release need attention, 2 open risk(s).", "status": "red", "milestones": [{"name": "GA release", "due": "Mar 9", "status": "on_track"}, {"name": "security flagged issues", "due": "Not fixed yet", "status": "done"}, {"name": "cutover rehearsal", "due": "Nov 21", "status": "done"}, {"name": "Pegasus ML fraud model", "due": "Nov 7", "status": "on_track"}], "risks": [{"risk": "vendor uncertain release timeline", "impact": "high", "owner": "Sam", "mitigation": "build a mock API to unblock testing"}, {"risk": "integration dependency on another team's release", "impact": "high", "owner": "Ben", "mitigation": "dedicate SRE support to stabilize env"}], "dependencies": ["security team to provision environments", "legal to finish identity provider config"], "decisions_needed": [], "next_steps": ["Lucia to set up sync with backend", "Sam to dedicate SRE support to stabilize env"]}
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
Merge the following notes into a single JSON program status report:

```json
{
  "summary": "Merge notes",
  "status": "green",
  "milestones": [
    {
      "name": "Falcon checkout redesign",
      "due": "Falcon checkout redesign",
      "status": "green",
      "milestones": [
        {
          "name": "Falcon checkout redesign",
          "due": "Falcon checkout redesign",
          "status": "green",
          "milestones": [
            {
              "name": "Falcon checkout redesign",
              "due": "Falcon checkout redesign",
              "status": "green",
              "milestones": [
                {
                  "name": "Falcon checkout redesign",
                  "due": "Falcon checkout redesign",
                  "status": "green",
                  "milestones": [
                    {
                      "name": "Falcon checkout redesign",
                      "due": "Falcon checkout redesign",
                      "status": "green",
                      "milestones": [
                        {
                          "name": "Falcon checkout redesign",
                          "due": "Falcon checkout redesign",
                          "status": "green",
                          "milestones": [
                            {
                              "name": "Falcon checkout redesign",
                              "due": "Falcon checkout redesign",
                              "status": "green",
                            
```
**Fine-tuned output**

```json
{"summary": "Falcon checkout redesign is off track; 0/3 milestones complete, 1 open risk(s).", "status": "red", "milestones": [{"name": "training rollout", "due": "Nov 13", "status": "on_track"}, {"name": "data quality in old system", "due": "Nov 13", "impact": "medium", "owner": "Fatima", "mitigation": "run data cleansing job before migration"}], "risks": [{"risk": "key engineer on leave", "impact": "high", "owner": "Carlos", "mitigation": "profile and optimize hot paths"}, {"risk": "key engineer on leave", "impact": "high", "owner": "Fatima", "mitigation": "run data cleansing job before migration"}], "dependencies": ["pricing strategy approval", "pricing strategy approval", "vendor license renewal"], "decisions_needed": ["off-site next test rollout", "pilot release", "pilot release"], "next_steps": ["Miriam to send updated timeline by Apr 27", "Mike to schedule another test env", "Elena to review the budget"])}
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
Mercury CRM Consolidation Standup Dec 21

Summary:

- perf numbers not meeting SLA (could kill the date) Wei owns mitigation = profile and optimize hot paths
- API freeze done ✅ (Dec 19)
- Aisha to update RAID log
- Omar to book uat sessions
- Coffee machine on 3rd floor broken again
- (notes taken quickly, sorry for typos)

Status:

- (notes taken quickly, sorry for typos)

Next steps:

- (notes taken quickly, sorry for typos)

Risks:

- (notes taken quickly, sorry for typos)

Dependencies:

- (notes taken quickly, sorry for typos)

Decisions needed:

- (notes taken quickly, sorry for typos)

Next steps:

- (notes taken quickly, sorry for typos)

Risks:

- (notes taken quickly, sorry for typos)

Dependencies:

- (notes taken quickly, sorry for typos)

Decisions needed:

- (notes taken quickly, sorry for typos)

Risks:

- (notes taken quickly, sorry for typos)

Dependencies:

- (notes taken quickly, sorry for typos)

Decisions needed:

- (notes taken quickly, sorry for typos)

Risks:

- (notes taken quickly, sorry for typos)

Dependencies:

- (notes taken quickly, sorry for typos)

Decisions needed:

- (notes taken quickly, sorry for typos)

Risks:

- (notes taken quickly, sorry for typos)

Dependencies:

- (notes taken quickly, sorry for typos)

Decisions needed:

- (notes taken quickly, sorry for typos)

Risks:

- (notes taken quickly, sorry for typos)

D
```
**Fine-tuned output**

```json
{"summary": "Mercury CRM consolidation is off track; 0/3 milestones complete, API freeze need attention, 2 open risk(s).", "status": "red", "milestones": [{"name": "API freeze", "due": "Dec 19", "status": "done"}, {"name": "beta launch", "due": "Apr 24", "status": "on_track"}, {"name": "finance load under control", "due": "Dec 19", "status": "on_track"}], "risks": [{"risk": "key engineer on leave", "impact": "high", "owner": "Lucia", "mitigation": "profile and optimize hot paths"}, {"risk": "key engineer on leave", "impact": "high", "owner": "Lucia", "mitigation": "cross-train a backup engineer"}], "dependencies": ["frontend sign-off on architecture"], "decisions_needed": [], "next_steps": ["Aisha to update RAID log", "Omar to book UAT sessions", "Wei to profile and optimize hot paths"]}
```
