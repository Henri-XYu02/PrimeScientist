Title: Concrete Plan — CoT in Planning Only

Objective
- Test whether chain-of-thought (CoT) prompting enables generalizable algorithmic reasoning vs. pattern-matching (default hypothesis: CoT does not induce a general planning algorithm).
- Compare Answer-Only (AO) vs. CoT prompting on Blocksworld planning only; treat Parity, String Manipulation, and Arithmetic as diagnostics (excluded from final conclusions).

Models and Budget

Runtime Fast Path (to avoid timeouts)
- Smoke-run first: `python experiments/cot_generalization/run_experiments.py --runs 3 --max_diffs 1 --tasks parity string --templates ID --conditions direct zeroshot_cot`.
- Prefer `batch_generate()` over per-prompt loops; send small batches (<= 8) with jittered backoff; keep per-batch sleep ~50–100 ms.
- Ramp only after smoke passes: `--runs 30` and `--max_diffs 3`; keep OOD templates gated until ID completes.
- Trim few-shot demos during diagnostics to <= 2 per task to curb token bloat.
- Persist partial results every ~100 calls to allow safe resume after interruptions.




Environment Setup & Imports
- Ensure `utils/__init__.py` exists so `from utils...` imports work.
- In each top-level script (e.g., runners/analyses) prepend project root to `sys.path`:
  `ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir));
   sys.path.insert(0, ROOT)` before `from utils import ...`.
- Add a preflight check step in the plan: `python -c "import sys; sys.path.insert(0, '.'); import utils.llm_inference; print('ok')"`.
- If LLM client supports role messages, pass system header via a `system` role; otherwise, prefix it to the user text as done in the runner.

- Models: `gpt-3.5-turbo`, `gpt-4-turbo` via `utils.llm_inference.LLMInference` and `batch_generate()`.
- API budget: ≤ 10,000 calls per model (hard cap enforced by scheduling).

Datasets (Programmatically Generated)
- General: All data are generated deterministically with fixed seeds, saved as JSONL with one example per line containing `id`, `split`, `task`, `input`, `target`, `meta`.
- Splits per task family:
  - Dev: 100 instances (no model calls required; for generator QA only).
  - ID-Test: 600 instances (200 Easy, 200 Medium, 200 Hard).
  - OOD-Hard: 300 instances (same size scale as ID but harder structure/operations).
  - OOD-Large: 300 instances (longer/larger than ID scale).
- Totals per task family: 1,300 generated; 1,200 used in evaluation (ID-Test + OOD-Hard + OOD-Large). Four task families ⇒ 4,800 evaluated instances.

Task Family 1 — Blocksworld Planning (name: blocksworld-v1)
- State: list of stacks; each stack is bottom→top (e.g., `[[B,A],[C]]` means A on B; C alone). Blocks named `A..Z`.
- Actions: `put_on(X,Y)` (move top block X onto clear block Y), `put_on_table(X)` (move top block X to empty table position). Only top/clear moves allowed. No illegal moves allowed by validator.
- Difficulty controls:
  - Easy: 3 blocks; at most one nontrivial stack; plan length target ≤ 4.
  - Medium: 4–5 blocks; arbitrary stacks; plan length target 5–10.
  - Hard: 5 blocks; entangled stacks; plan length target 8–14.
  - OOD-Hard: 5 blocks; goals require unstacking/re-stacking with interference (heuristic-unfriendly).
  - OOD-Large: 6–7 blocks; arbitrary stacks.
- Generator parameters:
  - Seeds per split: Dev 13, ID-Test 17, OOD-Hard 19, OOD-Large 23.
  - Instances sampled by: sample initial stacks uniformly then sample a random valid plan (length in target range) to produce the goal by forward-sim; discard if constraints unmet; record optimal length via BFS (cap 7 blocks).
- Targets: a valid action sequence reaching the goal.
- Validation: custom simulator checks preconditions and applies actions; success if final state equals goal; optional optimality gap = `len(plan) - optimal_length` when available.

Task Family 2 — Parity / State Tracking (name: parity-v1)
- Input: natural-language sequence of operations over a single binary state (e.g., coin initially Heads). Ops include: flip, keep, flip if previous was flip, etc.
- Difficulty controls by sequence length: Easy 10 ops, Medium 20, Hard 30; OOD-Hard: includes conditional ops composition at ID lengths; OOD-Large: 40 ops.
- Generator: sample ops from a fixed op set; deterministically simulate ground-truth final state.
- Target: final state (Heads/Tails) and canonical label `H`/`T`.

Task Family 3 — String Manipulation (name: strxform-v1)
- Input: instruction + list of words; operations include index extraction, case transform, concatenation, reversal, and join with delimiters; templates guarantee unambiguous results.
- Difficulty controls by words count and rule depth: Easy 3–5 words, Medium 6–8, Hard 9–10; OOD-Hard: same sizes with nested conditional rules; OOD-Large: 12 words.
- Generator: sample words from a fixed lexicon; apply composed operations to compute target string.
- Target: exact output string.

Task Family 4 — Multi-step Arithmetic (name: arith-v1)
- Input: parenthesized expressions over small integers using `+ - *` and optional division as integer floor when included; no division by zero.
- Difficulty: Easy depth 1–2, Medium 3–4, Hard 5; OOD-Hard: heavy parentheses and mixed ops at ID token counts; OOD-Large: longer token counts (e.g., 25–35 tokens).
- Generator: recursively build expression by depth; evaluate with exact integer arithmetic.
- Target: exact integer result.

Data Layout and Reproducibility
- Output directories:
  - `data/blocksworld-v1/{dev,id_test,ood_hard,ood_large}.jsonl`
  - `data/parity-v1/{dev,id_test,ood_hard,ood_large}.jsonl`
  - `data/strxform-v1/{dev,id_test,ood_hard,ood_large}.jsonl`
  - `data/arith-v1/{dev,id_test,ood_hard,ood_large}.jsonl`
- Each record fields:
  - `id`: `TaskName:Split:Index`
  - `task`: one of the four names above
  - `split`: `dev|id_test|ood_hard|ood_large`
  - `input`: task-specific natural language prompt content (without CoT suffix)
  - `target`: exact ground truth
  - `meta`: dict with generation params (e.g., blocks, stacks, ops, depth, optimal_length if known)

Prompting Conditions
- AO (Answer-Only): instruction + `input` + explicit answer schema. Require final line as `Final Answer: <value>`.
- CoT: instruction + `input` + `Let's think step by step.` Require final line as `Final Answer: <value or plan>`.
- For Blocksworld, the `Final Answer` must be a JSON array of actions, e.g., `["put_on(A,B)", "put_on_table(C)"]`.

Hyperparameters and Inference Settings
- Common: `max_tokens=512` (Blocksworld 768), `top_p=1.0`, `presence_penalty=0.0`, `frequency_penalty=0.0`.
- AO runs: `temperature=0.0`.
- CoT runs: `temperature=0.0`.
- CoT Self-Consistency diagnostic (CoT-SC): `temperature=0.7`, 10 samples per prompt (aggregate by majority vote for classification; by validator-first for Blocksworld: pick first valid plan if any; else majority by normalized actions).
- Stop sequences: none (postprocess by regex for `Final Answer:`).
- Batch size: 32 per `batch_generate()` call (or nearest supported).

Evaluation Metrics
- Parity / String / Arithmetic:
  - Accuracy = exact match of `Final Answer` with `target` (case-sensitive for strings; ints for arithmetic).
  - ID vs OOD accuracies reported separately.
- Blocksworld:
  - Plan validity rate = fraction that pass simulator and reach exact goal.
  - Optimality gap (when `optimal_length` available) and average plan length on valid plans.
  - ID vs OOD reported separately.
- Across tasks:
  - Compare AO vs CoT; CoT-SC reported on OOD-Hard only as a diagnostic.

API Call Accounting (per model)
- Instances evaluated: 4 tasks × (ID 600 + OOD-Hard 300 + OOD-Large 300) = 4,800.
- AO + CoT for all ⇒ 4,800 × 2 = 9,600 calls.
- CoT-SC diagnostic: OOD-Hard only, 10 instances per task × 10 samples = 100 per task ⇒ 400 calls.
- Total = 9,600 + 400 = 10,000 calls (within budget).

Step-by-Step Reproduction Procedure
1) Setup
   - Ensure Python 3.10+ and dependencies installed (jsonlines, numpy, tqdm). Make sure `utils/llm_inference.py` is available.
   - Confirm API access by instantiating `LLMInference(model_name)` once.

2) Generate Data
   - Run generator scripts to create JSONL files exactly as specified above with fixed seeds:
     - Blocksworld: seeds {dev:13, id_test:17, ood_hard:19, ood_large:23}; sizes per split as listed.
     - Parity: same seeds per split; lengths per difficulty as listed.
     - String: same seeds per split; words/rule depth per difficulty as listed.
     - Arithmetic: same seeds per split; depth/token counts per difficulty as listed.
   - Validate generation by re-computing each `target` from `meta` and asserting equality.

3) Define Prompts
   - AO template (classification-like):
     """
     You will be given a task. Provide only the final answer.
     Task: {input}
     Final Answer:
     """
   - AO template (Blocksworld):
     """
     You are given an initial and a goal configuration of blocks as stacks (bottom→top).
     Provide only a valid plan as a JSON array of actions using put_on(X,Y) or put_on_table(X).
     Task: {input}
     Final Answer:
     """
   - CoT template: same as AO plus the line: "Let's think step by step." before Final Answer.

4) Inference Scheduling (per model, per condition)
   - For each task family and each evaluated split (ID-Test, OOD-Hard, OOD-Large):
     - Run AO with `temperature=0.0` over all examples using `batch_generate()`; parse `Final Answer`.
     - Run CoT with `temperature=0.0` over all examples; parse final answer.
   - CoT-SC diagnostic: For OOD-Hard only, sample 10 examples per task; run 10 samples each with `temperature=0.7`; aggregate as specified.
   - Persist raw generations to `runs/{model}/{condition}/{task}/{split}.jsonl` with fields: `id`, `prompt`, `completion`, `parsed_answer`, `metadata`.

5) Evaluation
   - Parity/String/Arithmetic: compute accuracy by exact match.
   - Blocksworld: run simulator on parsed plan; mark valid/invalid; compute success rate, avg length on valid plans, and optimality gap where `optimal_length` exists.
   - Save metrics per task × split × condition × model to `metrics/metrics.tsv` with columns: `model,task,split,condition,n,primary_metric,secondary_metrics_json`.

6) Aggregation and Reporting

Reporting Guardrails (Do Not Overclaim)
Conclusion Scope (Planning Only)
- Limit conclusions to Blocksworld planning results; do not generalize to other algorithmic tasks.
- Do not make claims about arithmetic, string manipulation, or parity beyond reporting their own metrics; exclude them from the main conclusion.
- Banned phrases: "generalizable stepwise computation", "generalizes better across paraphrased instructions", "generalizes better than rigid few-shot pattern prompting".
Claim Gate for Planning
- If CoT success on OOD-Hard or OOD-Large < 70% OR drops > 10 points vs ID, explicitly conclude "CoT does not confer robust planning ability in this setup" and avoid any generalization claims.
Logging Hygiene
- Never print or store API keys, access tokens, or environment variables in logs; redact with '****' if needed.
- Report Blocksworld (planning) separately from synthetic tasks; do not conflate results across task families.
- Do not claim generalizable algorithmic reasoning for planning unless accuracy holds under OOD-Hard and OOD-Large and does not degrade with goal length/stack size.
- Always report ID vs OOD metrics separately; avoid pooled averages that mask generalization failures.
- Treat tightly patterned few-shot CoT as a potential overfitting baseline; do not use it as evidence of generality.
- When using algorithmic guidance, compare generic vs highly specific prompts; if performance drops with generic guidance, state this explicitly.

   - Produce tables summarizing AO vs CoT per model on ID-Test, OOD-Hard, OOD-Large for each task.
   - Include CoT-SC diagnostics on OOD-Hard.
   - Create plots (optional) for accuracy/validity vs. difficulty.

7) Determinism and Checks
   - Fix seeds as listed; do not shuffle order between AO and CoT; reuse identical `input` strings.
   - Enforce the 10,000 calls cap by counting planned calls before dispatch and halting if approaching the cap.

Mandatory Complexity Sweep (Blocksworld)
- Evaluate success rate as a function of block count n with strict coverage: n ∈ {3,4,5,6,7,8}.
- Per n and per split (ID-Test, OOD-Hard, OOD-Large), generate ≥ 30 instances (10 easy, 10 medium, 10 hard).
- Report a table of success@n and a plot of success vs n; fail the run if any n has < 30 evaluated instances.
- OOD-Large must include n ≥ 6 (6–8 blocks) — do not downscale due to runtime; instead cap plan length or prune via BFS depth limits.
- Conclusions about CoT generalization in planning must reference the success@n trend explicitly (claim-gating).

Validation Details Specific to Blocksworld
- Simulator rules:
  - A block is clear iff it is at the top of some stack (or alone on table).
  - `put_on(X,Y)` valid iff X is clear, Y is clear, X ≠ Y; removes X from its current stack, pushes onto Y’s stack.
  - `put_on_table(X)` valid iff X is clear; removes X from current stack, creates new stack `[X]`.
  - Goal satisfied iff stacks match exactly (order and grouping).
- Optimal length: BFS over states with branching per legal move; stop at goal; cap at 7 blocks.

Deliverables
- Data files in `data/…` as specified.
- Generation scripts and validators.
- Run logs and raw outputs in `runs/…`.
- Final metrics table `metrics/metrics.tsv` and a short report summarizing AO vs CoT outcomes (separate from this plan).



Conclusion Output Template (Planning-Only, Enforced)
- At the end of the report, emit exactly one line starting with `Planning-Only Conclusion:` chosen verbatim from the following options:
  - Planning-Only Conclusion: CoT does not confer robust planning ability in this setup.
  - Planning-Only Conclusion: CoT shows mixed planning performance and is not robust under OOD.
  - Planning-Only Conclusion: CoT planning performance is robust across ID and OOD settings.
- Do not add any other claims or qualifiers to this line. Do not mention arithmetic, string manipulation, or parity in this line.
- Immediately after this line, emit a JSON object on a new line with keys `task_family` (must be "blocksworld"), `id_acc`, `ood_hard_acc`, `ood_large_acc`, and `cot_condition` summarizing only Blocksworld metrics.
- Banned anywhere in the report (hard gate; fail the run if present): "generalizable algorithmic reasoning", "rule-based symbolic reasoning", "generalizes better across paraphrased instructions", "generalizable stepwise computation".
- Before writing the conclusion, run a string-level assertion that the banned phrases are absent; if any appear, replace the conclusion line with: `Planning-Only Conclusion: CoT does not confer robust planning ability in this setup.`
