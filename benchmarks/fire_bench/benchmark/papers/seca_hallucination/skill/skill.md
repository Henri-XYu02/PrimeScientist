> **INHERITED WORKSPACE** (score=0.400): The sandbox is seeded from the parent node's best research scripts. The skill below is the COMPLETE updated experimental plan; see `./INHERITED_FROM_PARENT/changes.md` for a short prose summary of what differs from the parent's plan and `./INHERITED_FROM_PARENT/diff.py` for the script that produced it. Update your scripts wherever the plan differs; you do not need to re-do steps the parent completed correctly.

Title: SECA Hallucination — Format/Noise Invariance Study (1-Hour Plan)

Task type: RESEARCH REPLICATION (variant of baseline)

Research question
- Do semantically neutral formatting and surface-noise perturbations (padding, wrappers, casing/markup) cause language models to hallucinate or produce inconsistent answers, despite the question text remaining unchanged?

Scope and resources (fixed)
- Dataset: TriviaQA, config unfiltered via datasets.load_dataset("trivia_qa", "unfiltered"). Use the validation split. Filter to short, factoid answers (aliases with max 3 tokens after normalization).
- Models (primary): gpt-4o-mini, gpt-4.1 via utils.llm_inference.LLMInference.
- Optional HF models: meta-llama/Llama-3.1-8B-Instruct; Qwen/Qwen2.5-7B-Instruct; meta-llama/Llama-2-13b-chat-hf.
- Inference utility: LLMInference(...). Use batch_generate() for throughput. API keys are wired by the harness.
- Budget: <=1000 API calls per model.
- Outputs: per-model TSV (append to results.tsv) and per-run JSONL in runs/{timestamp}_{model}.jsonl.

Definitions (kept from baseline)
- Inconsistency: for a base question, at least two semantically equivalent perturbations yield different normalized answers. Majority Agreement Rate (MAR) = max_variant_count / num_variants.
- Hallucination: an answer that is wrong (Exact Match = 0) and presented confidently. Confidence heuristic: (a) no hedging markers from the fixed list; (b) answer length <= 8 tokens.

Additional definition (new)
- Format Sensitivity Score (FSS): 1 - MAR, computed only over formatting/noise variants that keep the raw question text byte-identical inside the wrapper. Higher FSS indicates greater sensitivity to formatting-only changes.

Evaluation metrics (exact)
- Exact Match (EM) with SQuAD-style normalization.
- Majority Agreement Rate (MAR); Flip Rate across variants.
- Hallucination Rate (HR) under the confidence heuristic.
- New: Format Sensitivity Score (FSS) and per-perturbation accuracy/hallucination breakdown.

Experiment design (format/noise focus)
- Base sample: N_base = 200 validation questions, deterministic seed=20260429. Filters: question length 6–25 tokens; at least one gold alias length <= 3 tokens (normalized).
- Variants per question: V = 8 (1 original + 7 format/noise perturbations) that DO NOT alter the question string itself, only its surroundings or presentation.
- Total prompts per model: 200 x 8 = 1600.

Exact perturbation set (apply in order; dedupe if identical)
1) orig: the raw question as user content.
2) md_heading: Prepend "# Question" then a blank line, then the question.
3) xml_wrap: Wrap the question in a neutral XML tag: "<question>...Q...</question>".
4) code_fence: Place the question inside triple backticks with a neutral language label: "```txt
Q
```" and then add: "Answer: " on a new line.
5) left_padding: Prepend 120 space characters before the question (single line).
6) boilerplate_note: Prepend a neutral note line "Note: The following is a question." then the question.
7) emoji_bookends: Add a harmless emoji prefix/suffix: "📘 Q 📘" (the inner Q is the exact question).
8) case_title: Present the question in Title Case (letters only), but also include the original question verbatim below, separated by a newline, and instruct: "Use the original question only." This keeps the answer intent while probing presentation changes.

Prompt template
- user message: exactly the variant text above.
- No system prompt. No few-shot history.
- Expected answer: short span with only the answer; punctuation optional.

Hyperparameters (fixed)
- max_tokens=24, temperature=0.2, top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0.
- batch_generate: batch_size=100, concurrency=8, request_timeout_s=30, retry=3 (exponential backoff).
- Per-model cap: 1800 prompts including retries.

Normalization (unchanged from baseline)
- Lowercase; strip punctuation [^a-z0-9\s]; remove articles {a, an, the}; normalize whitespace; ordinal mapping up to 20th; simple plural singularization when gold set contains singular.

Procedure
1) Load data: datasets.load_dataset("trivia_qa", "unfiltered")["validation"], seed=20260429; filter as specified.
2) Build perturbations implementing the 8 variants; ensure the raw question text inside wrappers is byte-identical for variants 2–7. For variant 8, include both title-cased and original lines and instruct to use the original.
3) Prepare prompts and log qid, variant_id, model, and prompt text.
4) Run inference per model with the fixed hyperparameters; save raw JSONL per model.
5) Normalize outputs and score EM; flag hallucination via the heuristic; compute MAR, Flip Rate, per-perturbation metrics.
6) Compute FSS: for each qid, restrict to variants {md_heading, xml_wrap, code_fence, left_padding, boilerplate_note, emoji_bookends} plus orig; FSS = 1 - MAR over this subset.
7) Summarize per model: EM, HR, MAR, Flip Rate, FSS; include top 10 examples with highest FSS and any hallucination examples.
8) Append a TSV row to results.tsv with columns: model	N_base	V	EM	MAR	FlipRate	HR	FSS	seed	timestamp.

Run profile (1 hour)
- 1600 prompts per model; batches of 100 with concurrency 8 complete within ~1 hour for the API models.

Unspecified decisions (fixed)
- No system prompt; single-turn user message only.
- If any wrapper would truncate the question or reflow entities, skip and backfill with md_heading to keep V constant; keep the intended label.

Deliverables
- Updated results.tsv; raw runs in runs/; concise narrative noting whether formatting-only noise triggers inconsistencies or hallucinations with concrete IDs.

Differences vs parent (critical)
- Replaces instruction-style phrasing perturbations with strictly formatting/noise wrappers that keep content identical; adds Format Sensitivity Score and reporting.
