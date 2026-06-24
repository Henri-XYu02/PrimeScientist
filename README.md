# MetaScientist: Budget-Adaptive Research Plan Tree Search

Code for **MetaScientist**, a budget-adaptive tree-search framework for
autonomous research agents. Each tree node is an *experimental plan*
executed by a coding agent; a *reflector agent* reads each result and
proposes child plans. A budget-adaptive selection rule (BAVT,
`α = 1/r_t`) shifts the search from exploration to exploitation as the
token budget is consumed.

Evaluated on three benchmarks:
- **FIRE-Bench** — research-paper replication, scored by RAGChecker F1.
- **AutoLab** — systems-engineering code optimization, scored by throughput ratio.
- **MLE-Bench** — Kaggle-style ML engineering.

Baselines: a **linear AutoResearch** edit-run-keep-or-revert loop
(`run_vanilla.py`) and a single-shot agent — both under the same token budget.

> Paper: *MetaScientist*.
> arXiv link forthcoming.

---

## Quick start

```bash
# 1. Conda env
conda create -n metascientist python=3.11 -y
conda activate metascientist
pip install -r requirements.txt

# 2. Secrets
cp .env.example .env       # fill in OPENAI_API_KEY etc.

# 3. (FIRE-Bench only) build the per-run docker image
docker build -t firebench-codex:0.1 benchmarks/fire_bench/agents/codex/

# 4. Run
bash scripts/run_firebench.sh   # tree search over the FIRE-Bench suite
bash scripts/run_autolab.sh     # tree search over the AutoLab suite
tmux attach -t firebench        # watch
tmux attach -t autolab
```

See [`scripts/README.md`](scripts/README.md) for the full launcher set,
flag reference, and how to customize the task list. Smoke-test by setting
`--budget 1` (search) or `--max_iters 1` (vanilla) in the launcher's `COMMON` line.

---

## Repository layout

```
run_search.py               # entry point: tree search (MetaScientist)
run_vanilla.py              # entry point: linear AutoResearch baseline
scripts/                    # tmux launchers for FIRE-Bench / AutoLab
  run_firebench.sh
  run_firebench_vanilla.sh
  run_autolab.sh
  run_autolab_vanilla.sh

search/
  tree.py                   # TreeNode + BAVT/UCT/Greedy/Random selection
  reflector.py              # reflector agent (codex or claude)

benchmarks/
  base.py                   # Benchmark ABC
  registry.py               # name → Benchmark class
  utils.py                  # log parsing, cost pricing, ledger I/O
  fire_bench/               # FIRE-Bench (research-paper replication)
    agents/                 # per-agent runners (codex, claude, openhands)
    benchmark/papers/       # per-task instruction + RAGChecker data
    eval/RAGChecker/        # vendored RAGChecker evaluator
    utils/                  # shared helpers copied into each sandbox
  autolab/                  # AutoLab benchmark (vendored from autolabhq/autolab
                            # at 6c7968d + our task-instruction edits)
```

`fire_bench_data/` (override with `$FIRE_BENCH_DATA`), `search_tree/`
(`$FIRE_BENCH_SEARCH_TREE`), and `vanilla_runs/` (`$FIRE_BENCH_VANILLA_RUNS`)
are created at runtime and gitignored.

---

## Configuration

Read from `.env` via `python-dotenv`. Full list documented inline in
[`.env.example`](.env.example):

| Variable | What | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI access for codex agent + reflector | required |
| `ANTHROPIC_API_KEY` | Anthropic access for claude agent | if using claude |
| `HF_TOKEN` | Hugging Face token (some FIRE-Bench tasks) | if using HF |
| `GOOGLE_API_KEY` | Gemini (some FIRE-Bench tasks) | optional |
| `FIRE_BENCH_DATA` | Per-run logs / sandboxes / RAGChecker output dir | `./fire_bench_data` |
| `FIRE_BENCH_SEARCH_TREE` | Override `--output_dir` for `run_search.py` | `./search_tree` |
| `FIRE_BENCH_VANILLA_RUNS` | Override `--output_dir` for `run_vanilla.py` | `./vanilla_runs` |
| `FIREBENCH_USE_DOCKER` | `1` to run the FIRE-Bench codex agent inside docker | `0` |

---

### Setup Codex

FIRE-Bench uses codex 0.39.0 by default, you need to fill in your OPENAI_API_KEY manually in .codex/auth.json for FIRE-Bench experiments.


## Pipeline

```
       Reflector drafts root skill (plan)
              │
              ▼
        Agent executes skill ──► evaluator scores result
              │                  (RAGChecker F1 / AutoLab reward)
              ▼
        Reflector reads score + log ──► proposes m child plans
              │
              └── BAVT selects next node; repeat until budget exhausted
```

BAVT selection rule:

```
P(v | u) ∝ w(v)
w(v) = Q(v)^α_t                       if v has been evaluated
w(v) = (Q(u) · √P(v))^α_t             if v is unvisited
α_t  = min(1/r_t, α_max),   r_t = remaining_budget / B
```

As `r_t → 0` the distribution sharpens toward `argmax Q`, automatically
shifting exploration → exploitation.

---

## Reproducibility notes

- **Backbone:** GPT-5 via Codex CLI (`codex@0.121.0`).
- **Per-task token budget:** `B = 1.5 × 10⁶` (configurable via `--max_tokens`).
- **Hyperparameters:** `m = 3` proposals per expansion, `α_max = 10`,
  prune threshold `δ = 0.05`. Exact flags in [`scripts/`](scripts/).
- **High-variance FIRE-Bench tasks** are evaluated twice per plan; the budget
  tracker charges both runs.
- **AutoLab** verifier runs in Docker with per-task resource limits from
  `tasks/{task}/task.toml`.

