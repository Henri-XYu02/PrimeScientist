# Reflect Program

This file defines the rules and constraints for the CodeAct reflector.
Edit it to tune reflection behaviour without touching code.

---

## What you ARE allowed to do

- **global_skill.md**: Add or sharpen universal, process-level guidelines
  (e.g. "Verify that the loaded dataset matches the expected schema before running experiments").
  Use exactly five sections:
  `Experimental Rigor | Staying on Task | Termination & Reporting | Implementation Soundness | Autonomy & Constraints`
- **global_errors.md**: Append CONCRETE error patterns observed in the logs
  (e.g. "[Method Deviation] Agent used synthetic Gaussian data instead of the provided CSV dataset").
  Use the error-type prefix from the taxonomy where applicable.
- **Per-task skill.md**: Update the experimental plan — fix broken implementation steps,
  adjust data budget based on observed timing, add pitfall warnings grounded in the logs.
- You can ADD, EDIT, REMOVE any entries in these files.

## What you must NOT do

- Remove sections or entries that appear to have worked correctly.
- Change the Research Question or the Evaluation Metrics section in per-task skill files.
- Speculate beyond what is directly evidenced in the provided logs and scores.
- Make the plan longer than necessary — prefer targeted, high-confidence fixes.

## File size discipline

The FILE SIZES section shows the current size of each skill file with a guidance label:
- **✓ good size** (< 10 KB): add freely, but still prefer precision over length.
- **⚠ getting long** (10–20 KB): edit or replace existing entries rather than appending new ones.
  Consolidate related entries; remove redundant or low-value content.
- **✗ too large** (> 20 KB): you MUST trim before adding anything.
  Merge duplicate guidelines, drop entries that are no longer evidenced, shorten verbose prose.

The injected skill is part of every agent's prompt context. Every extra kilobyte competes for
the agent's attention. Shorter, denser skill files outperform longer, diffuse ones.

## Scoring interpretation

RAGChecker metrics (all in 0–100 range, higher is better):
- **Precision**: fraction of agent's claims that are correct. Low → agent made false/extra claims.
- **Recall**: fraction of ground-truth claims captured. Low → agent missed key findings.
- **F1**: harmonic mean — primary optimisation target.

When scores are low, look for the root cause in the logs before editing the plan.

## Keep / discard awareness

The workflow compares per-task F1 across consecutive rounds.
If a task's F1 **degraded** after your update, the system will **revert** your changes
and re-run reflection with the reverted base.
Therefore: **make only high-confidence, evidence-based edits**.
Speculative or overly broad changes are more likely to trigger a revert.

## Data budget rules (per-task skill)

| Observed timing          | Action                                      |
|--------------------------|---------------------------------------------|
| Any run > 110 min        | Decrease sample size; add ⚠ timeout note   |
| All runs < 15 min        | Increase sample size unless already using full dataset     |
| Runs within 15–110 min   | Keep sample size; adjust only if F1 is poor |
