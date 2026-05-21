## Research Question
Measure, by running end-to-end experiments, the accuracy improvement from Chain-of-Thought prompting over Direct Answer prompting across Commonsense, Knowledge, Symbolic, Mathematical, and Soft Reasoning categories, and determine which category-level gains are statistically significant.

## Dataset & Data Budget
- Exact dataset source
  - Mathematical: GSM8K via `load_dataset("openai/gsm8k", "main")` or `/data/gsm8k/`
  - Commonsense: CommonSenseQA via `load_dataset("tau/commonsense_qa")` or `/data/commonsenseqa/`
  - Soft Reasoning: Winogrande via `load_dataset("allenai/winogrande", "winogrande_debiased")`
  - Symbolic + Knowledge: FOLIO via `load_dataset("yale-nlp/FOLIO")` or `/data/folio/`
- Category mapping to datasets
  - Commonsense → CommonSenseQA
  - Soft Reasoning → Winogrande
  - Mathematical → GSM8K
  - Symbolic → FOLIO examples whose evaluation target can be reduced to formal validity/entailment with SALM
  - Knowledge → FOLIO examples evaluated as factual/logical world-knowledge style entailment; operationally, use the same sampled FOLIO set and report FOLIO as both Symbolic and Knowledge only if the dataset includes enough examples to split by metadata; otherwise report FOLIO under Symbolic/Knowledge combined and state that this is the closest available dataset under the provided resources
- Recommended sample size for 2-hour budget
  - Keep exactly 60 evaluation examples per dataset
  - Total examples: 4 datasets × 60 = 240
  - Total prompt conditions per model: 4
  - Total models: 3
  - Total inference jobs: 240 × 4 × 3 = 2880 model calls
  - Recent runs remained within the allowed 15–110 min window, including one much faster run that achieved poor plan fidelity, so do not reduce the sample size for speed; prioritize implementation correctness and exact protocol matching instead
- How to sample
  - Use official held-out splits only; never evaluate on ad hoc local merged files unless they are verified to correspond to the intended split
  - For this task, local fallback files such as generic `gsm8k.jsonl`, `commonsenseqa.json`, or `folio/original.json` must not be used for evaluation unless you have first established and recorded that they exactly match the official split provenance and schema; otherwise stop and repair the official loader
  - Random seed: `2025`
  - Sampling strategy
    - GSM8K: sample 60 from `ds["test"]` uniformly without replacement
    - CommonSenseQA: sample 60 from `ds["validation"]` uniformly without replacement
    - Winogrande: sample 60 from `ds["validation"]` uniformly without replacement
    - FOLIO: sample 60 from the official validation split if present, else test/dev split available from the dataset object, uniformly without replacement
  - Persist both sampled example ids/indices and the source split names to `artifacts/sample_indices.json` so reruns are identical and auditable
- Train / validation / test split
  - No training
  - Use held-out evaluation splits only
  - Few-shot demonstrations must be drawn from the training split of the same dataset, with fixed demonstration indices disjoint from the 60 evaluation examples
  - If a dataset cannot be loaded from its official train split because of access or cache issues, stop and repair the loader rather than silently substituting evaluation examples as demonstrations
  - Use 4 demonstrations for every few-shot condition

## Experimental Conditions
Enumerate all required conditions exactly as follows.

1. Models
   - `gpt-4o`
   - `claude-3-5-sonnet-20240620`
   - `mistralai/Mistral-7B-Instruct-v0.3`

2. Prompt variants
   - Zero-shot Direct Answer
   - Zero-shot CoT
   - Few-shot Direct Answer
   - Few-shot CoT

3. Datasets / reasoning categories
   - GSM8K → Mathematical
   - CommonSenseQA → Commonsense
   - Winogrande → Soft Reasoning
   - FOLIO → Symbolic / Knowledge as available from dataset metadata; if no metadata split is available, report as combined Symbolic+Knowledge and explicitly mark the limitation

4. Required inference/evaluation method by dataset
   - GSM8K
     - Generate reasoning or answer according to prompt condition
     - Use Program-aided Language Model evaluation only as a post-processing aid: first extract fenced Python blocks or a clearly delimited final expression, then execute only that extracted snippet in a hardened sandbox
     - Do not execute the entire raw model response as Python
     - If no executable snippet is present, fall back to robust numeric-answer extraction from the final answer line
   - FOLIO
     - Use Satisfiability-Aided Language Model evaluation with Z3 only when the model explicitly emits a structured formalization in a predefined format that your parser recognizes
     - Parse model output into one of `entailment`, `contradiction`, `unknown`
     - Do not execute arbitrary free-form model text as Z3/Python code; if no trusted formalization is present, use label extraction fallback
   - CommonSenseQA
     - Standard multiple-choice evaluation, exact match on option letter or normalized option text
   - Winogrande
     - Binary-choice evaluation, exact match on option `1`/`2`

5. Fixed decoding parameters for all models and all conditions
   - Temperature: `0.0`
   - Top-p: `1.0`
   - Max tokens:
     - GSM8K CoT: `512`
     - GSM8K Direct: `128`
     - FOLIO CoT: `384`
     - FOLIO Direct: `96`
     - CommonSenseQA CoT: `192`
     - CommonSenseQA Direct: `48`
     - Winogrande CoT: `128`
     - Winogrande Direct: `32`

6. Number of samples per condition
   - Exactly 60 evaluation examples per dataset per model per prompt variant
   - Exactly 4 few-shot demonstrations per few-shot condition

7. Baseline and comparison
   - Baseline for each category/model/prompt-family comparison: corresponding Direct Answer condition
   - Main comparison:
     - Zero-shot CoT vs Zero-shot Direct
     - Few-shot CoT vs Few-shot Direct
   - Secondary comparison:
     - Best CoT condition vs best Direct condition within each model-category pair

## Implementation Specification
- Architecture or model name
  - Use `LLMInference` for:
    - `gpt-4o`
    - `claude-3-5-sonnet-20240620`
    - `mistralai/Mistral-7B-Instruct-v0.3`
- Optimizer
  - None; no model training
- Batch size, number of epochs / steps, early-stopping criterion
  - No training epochs
  - Inference batch size: `1` request per example; run asynchronously with concurrency `8` per model if supported
  - Retry failed API calls up to `2` times with exponential backoff `2s, 6s`
  - Degenerate-condition early stop: if the first 15 examples of a condition have parseable prediction rate `< 0.2`, stop the remaining examples for that condition, mark it failed, and continue with other conditions
- Random seed
  - `2025` for sampling, demonstration selection, and any randomized ordering
- Pre-run implementation gates
  - Before launching the full grid, run a 2-example smoke test for every dataset/prompt family to verify: dataset loader works, prompt builder returns the exact required template, parser returns a valid prediction on at least 1 of the 2 examples, and artifact paths are writable
  - Fail fast if any loader falls back to a non-official local file with unknown split provenance
  - For every dataset loader, print and save the resolved dataset name, configuration, split name, item count, and a 1-example schema snapshot to `artifacts/loader_audit.json` before inference starts
  - Fail fast on syntax errors in helper modules before spending API calls; run a Python compile/import check over all helper modules first
  - Validate that the actual `LLMInference` call signature matches the planned arguments on one dry-run request per provider

- Exact prompt templates

  - Zero-shot Direct Answer, CommonSenseQA
    - System: `You are a careful reasoning assistant. Answer with only the single best option letter: A, B, C, D, or E.`
    - User:
      - `Question: {question}`
      - `Options:`
      - `A. {A}`
      - `B. {B}`
      - `C. {C}`
      - `D. {D}`
      - `E. {E}`
      - `Answer:`

  - Zero-shot CoT, CommonSenseQA
    - System: `You are a careful reasoning assistant. Think step by step, then end with a line exactly in the format Final Answer: X where X is one of A, B, C, D, or E.`
    - User: same question/options block

  - Few-shot Direct Answer, CommonSenseQA
    - System: same as zero-shot direct
    - User:
      - 4 fixed demonstrations in the format:
        - `Question: ...`
        - `Options: ...`
        - `Answer: X`
      - Then target question block
    - Demonstration indices: first 4 sampled from training split with seed `2025`, excluding eval overlaps

  - Few-shot CoT, CommonSenseQA
    - System: same as zero-shot CoT
    - User:
      - 4 fixed demonstrations in the format:
        - `Question: ...`
        - `Options: ...`
        - `Reasoning: ...`
        - `Final Answer: X`
      - Then target question block

  - Zero-shot Direct Answer, Winogrande
    - System: `Choose the correct option. Reply with only 1 or 2.`
    - User:
      - `Sentence: {sentence_with_blank}`
      - `Option 1: {option1}`
      - `Option 2: {option2}`
      - `Answer:`

  - Zero-shot CoT, Winogrande
    - System: `Think step by step about which option best fits the sentence. End with a line exactly in the format Final Answer: X where X is 1 or 2.`
    - User: same content

  - Few-shot Direct Answer, Winogrande
    - Same structure with 4 training demonstrations ending `Answer: 1/2`

  - Few-shot CoT, Winogrande
    - Same structure with 4 training demonstrations ending `Final Answer: 1/2`

  - Zero-shot Direct Answer, GSM8K
    - System: `Solve the problem. Reply with only the final numeric answer.`
    - User: `Problem: {question}`

  - Zero-shot CoT, GSM8K
    - System: `Solve the problem step by step. If useful, write short Python code. End with a line exactly in the format Final Answer: <number>.`
    - User: `Problem: {question}`

  - Few-shot Direct Answer, GSM8K
    - 4 training demonstrations:
      - `Problem: ...`
      - `Answer: <number>`

  - Few-shot CoT, GSM8K
    - 4 training demonstrations:
      - `Problem: ...`
      - `Reasoning: ...`
      - `Python: ...` if available or handcrafted concise arithmetic steps
      - `Final Answer: <number>`

  - Zero-shot Direct Answer, FOLIO
    - System: `Determine whether the conclusion follows from the premises. Reply with only one label: entailment, contradiction, or unknown.`
    - User:
      - `Premises: {premises}`
      - `Conclusion: {conclusion}`
      - `Label:`

  - Zero-shot CoT, FOLIO
    - System: `Reason step by step about whether the conclusion is logically implied, contradicted, or neither. End with a line exactly in the format Final Answer: LABEL where LABEL is entailment, contradiction, or unknown.`
    - User: same premises/conclusion block

  - Few-shot Direct Answer, FOLIO
    - 4 training demonstrations ending `Label: entailment|contradiction|unknown`

  - Few-shot CoT, FOLIO
    - 4 training demonstrations ending `Final Answer: entailment|contradiction|unknown`

- Prompt-template compliance check
  - Implement prompts as structured `system` and `user` messages matching the exact text above, not paraphrased single-string approximations
  - Before the full run, save one rendered prompt per dataset × variant to `artifacts/prompt_audit/` and verify by string comparison that the templates exactly match the plan except for field substitution and the explicit degenerate-rerun reminder
  - Do not introduce extra methods or renamed variants such as PAL-only or Z3-code-generation conditions as substitutes for the four required prompt variants

- API call pattern
  - For each model in `[gpt-4o, claude-3-5-sonnet-20240620, mistralai/Mistral-7B-Instruct-v0.3]`
  - For each dataset in `[gsm8k, commonsenseqa, winogrande, folio]`
  - For each prompt variant in `[zs_direct, zs_cot, fs_direct, fs_cot]`
  - For each sampled example index
    - Call `LLMInference.generate(model=..., prompt=..., temperature=0.0, top_p=1.0, max_tokens=...)`
    - Save raw response to `artifacts/raw/{model}/{dataset}/{prompt_variant}.jsonl`
    - Parse prediction and save to `artifacts/parsed/...`
  - After each condition, compute rolling parse rate and accuracy snapshot

- Statistical / data-analysis tasks
  - Primary metric per condition:
    - Accuracy = `(1/n) * sum_i 1[pred_i = gold_i]`
  - Improvement for a matched comparison:
    - `Delta = Accuracy(CoT) - Accuracy(Direct)`
  - Significance test within each model-category pair and prompt-family pair
    - Use McNemar’s exact test on paired correctness outcomes over the same 60 examples
    - Let
      - `b = # examples correct under Direct and wrong under CoT`
      - `c = # examples wrong under Direct and correct under CoT`
    - Two-sided exact p-value on `min(b,c)` under Binomial(`n=b+c`, `p=0.5`)
    - Do not replace this primary test with chi-squared approximation, exponential approximation, or an uncorrected asymptotic shortcut; if you report any approximation, label it auxiliary only
  - Aggregate significance by category across models
    - Pool paired outcomes across the 3 models by summing `b` and `c`, then run the same McNemar exact test
    - Also report mean delta across the 3 models
  - Multiple comparisons
    - There are up to 5 categories × 2 prompt-family comparisons = 10 primary category-level significance tests
    - Apply Benjamini-Hochberg FDR correction with `q = 0.05`
    - Also report uncorrected p-values
  - Confidence intervals
    - For each accuracy, compute Wilson 95% CI
    - For each delta, compute paired bootstrap 95% CI with `2000` resamples and seed `2025`

## Evaluation Protocol
- Primary metric
  - Accuracy on each dataset/category under each model and prompt variant
  - Library/formula:
    - `accuracy_score(y_true, y_pred)` from scikit-learn or exact equivalent manual computation
- Secondary metrics
  - Parseable prediction rate = fraction of outputs successfully mapped to valid labels/answers
  - For GSM8K:
    - Execution success rate of PAL step = fraction where generated code/expression executed successfully
  - For FOLIO:
    - Z3-usable formalization rate if a formal specification is attempted
  - Mean token usage per response if available from API metadata
- How to aggregate across conditions
  - Per model × category:
    - Report 4 accuracies
    - Report 2 main deltas:
      - `zs_cot - zs_direct`
      - `fs_cot - fs_direct`
  - Per category across models:
    - Mean accuracy for each prompt variant
    - Mean delta and pooled McNemar p-value
  - Overall:
    - Macro-average delta across categories separately for zero-shot and few-shot
- What constitutes a valid result vs a degenerate run
  - Valid condition:
    - Parseable prediction rate `>= 0.8`
    - Accuracy not trivially zero on all models for the dataset
  - Degenerate run:
    - Parseable prediction rate `< 0.8`, or
    - More than 20% API failures after retries, or
    - For GSM8K, numeric extraction fails on >50% of samples, or
    - For FOLIO, labels cannot be mapped for >20% of samples
  - If a condition is degenerate, rerun once with the same seed and a stricter answer-format reminder appended to the user prompt; if still degenerate, exclude from significance testing and report failure explicitly

## Expected Timeline
- Data loading & preprocessing: 12 min
  - Load datasets, sample 60 per dataset, choose 4 demos per dataset, build prompts, sanity-check parsers
- Main experiment / training / inference: 78 min
  - 48 conditions total = 3 models × 4 datasets × 4 prompts
  - Average target: about 1.6 min per condition with async concurrency and conservative sample sizes
- Evaluation & aggregation: 16 min
  - Parse outputs, run PAL/Z3 post-processing, compute accuracies/CIs/tests, generate tables and conclusions
- Total: 106 min

## Pitfalls Observed in Prior Run
- A prior run drifted from the specified prompt templates and dataset handling; keep prompts exactly as written in this plan unless invoking the explicit degenerate-condition rerun rule.
- Do not replace exact McNemar testing with chi-squared approximation for the primary significance analysis; compute the two-sided exact binomial p-value from discordant pairs, and use any asymptotic statistic only as an auxiliary diagnostic.
- Do not silently downgrade category coverage: Winogrande must be included for Soft Reasoning, and FOLIO must be reported as Symbolic/Knowledge combined if no metadata split is available.
- Avoid partial local loaders that read generic json/jsonl files without confirming schema and split identity against the official dataset specification.
- If sandbox helpers for PAL or SALM raise syntax/safety issues, disable only the unsafe execution path and continue with the specified extraction fallback rather than crashing the whole run.
- Save raw responses, parsed predictions, and per-condition summaries incrementally so a mid-run failure does not erase completed conditions.
- Prior runs incorrectly introduced renamed conditions and alternative prompt wording; keep the four required variants exactly as enumerated and do not substitute PAL/Z3 generation for CoT vs Direct comparisons.
- Prior runs attempted to use unofficial local GSM8K/CommonSenseQA/FOLIO files after loader friction; treat loader repair as mandatory, not optional.
- A prior run failed because a helper sandbox module contained syntax-corrupted code; add a compile/import gate before any API calls.

## Implementation Notes
- Do not pass extra arguments like `cache_dir` to `load_dataset("allenai/winogrande", "winogrande_debiased")`; use exactly that signature.
- Keep few-shot demonstrations fixed across models and prompt reruns; otherwise paired significance tests become less interpretable.
- For GSM8K, normalize answers by extracting the final numeric value, removing commas, currency symbols, and trailing periods before comparison.
- For CommonSenseQA and Winogrande, many models may output extra text despite direct instructions; implement strict regex extraction for final option labels.
- For FOLIO, use Z3 only as an aid when a formalizable structure is present; always keep a robust fallback that extracts one of `entailment`, `contradiction`, or `unknown` from the final answer line.