# MetaScientist: Budget-Adaptive Research Plan Tree Search

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

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

> Paper: *MetaScientist*. arXiv link forthcoming.

---

## Quick start

```bash
# 1. Environment
conda create -n metascientist python=3.11 -y
conda activate metascientist
pip install -r requirements.txt

# 2. Secrets
cp .env.example .env                       # fill in OPENAI_API_KEY etc.
cp .codex/auth.json.example .codex/auth.json   # fill in the same OpenAI key

# 3. (FIRE-Bench only) build the per-run docker image
docker build -t firebench-codex:0.1 benchmarks/fire_bench/agents/codex/

# 4. Run
bash scripts/run_firebench.sh   # tree search over the FIRE-Bench suite
bash scripts/run_autolab.sh     # tree search over the AutoLab suite
tmux attach -t firebench        # watch
```

Both `.env` and `.codex/auth.json` are gitignored — only the `.example`
files are tracked.

See [`scripts/README.md`](scripts/README.md) for the full launcher set, the
flag reference for both entry points, how to customize the task list, and
where the token budget comes from.

**Smoke test** before spending a full budget:

```bash
python run_search.py --benchmark autolab --task gaussian_blur --budget 1
```

---

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

## Repository layout

```
run_search.py               # entry point: tree search (MetaScientist)
run_vanilla.py              # entry point: linear AutoResearch baseline

scripts/                    # tmux launchers — see scripts/README.md
  run_firebench.sh          #   FIRE-Bench, tree search
  run_firebench_vanilla.sh  #   FIRE-Bench, baseline
  run_autolab.sh            #   AutoLab, tree search
  run_autolab_vanilla.sh    #   AutoLab, baseline

search/
  tree.py                   # TreeNode + BAVT/UCT/Greedy/Random selection
  reflector.py              # reflector agent (codex or claude)

benchmarks/
  base.py                   # Benchmark ABC
  registry.py               # name → Benchmark class
  utils.py                  # log parsing, cost pricing, ledger I/O
  fire_bench/               # FIRE-Bench (research-paper replication)
    agents/                 #   per-agent runners (codex, claude, openhands)
    benchmark/papers/       #   per-task instruction + RAGChecker data
    eval/RAGChecker/        #   vendored RAGChecker evaluator
    utils/                  #   shared helpers copied into each sandbox
  autolab/                  # AutoLab, vendored from autolabhq/autolab at
                            # 6c7968d plus our task-instruction edits

LICENSE                     # MIT
NOTICE                      # licenses / provenance of vendored components
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
| `CODEX_AUTH_DIR` | Host dir with codex credentials, mounted into the container | `~/.codex` |

### FIRE-Bench data directory (`FIRE_BENCH_DATA`)

FIRE-Bench keeps all per-run state (agent sandboxes, logs, RAGChecker results)
under one directory. Two options:

- **Leave it unset** — defaults to `<repo>/fire_bench_data`, created on first run.
- **Set it in `.env`** — point it at an absolute path on a bigger disk, e.g.
  `FIRE_BENCH_DATA=/scratch/$USER/fire_bench_data`.

The harness and the agent runners resolve this variable identically; if they
ever disagree, the agent writes logs where the harness never looks and every
run is reported as failed while still spending tokens.

---

## Reproducibility notes

- **Backbone:** GPT-5 via the Codex CLI. The two suites pin different
  releases: FIRE-Bench uses `@openai/codex@0.39.0` (baked into the docker
  image), AutoLab uses `@openai/codex@0.121.0` (fetched with `npx`).
- **Per-task token budget:** `B = 1.5 × 10⁶` (`--max_tokens`); derivation in
  [`scripts/README.md`](scripts/README.md).
- **Hyperparameters:** `m = 3` proposals per expansion, `α_max = 10`,
  prune threshold `δ = 0.05`. Exact flags in [`scripts/`](scripts/).
- **High-variance FIRE-Bench tasks** are evaluated twice per plan; the budget
  tracker charges both runs.
- **AutoLab** verifier runs in Docker with per-task resource limits from
  `benchmarks/autolab/tasks/{task}/task.toml`.

---

## Citation

```bibtex
@misc{metascientist,
  title  = {MetaScientist: Budget-Adaptive Research Plan Tree Search},
  author = {Yu, Xinle},
  year   = {2026},
  note   = {arXiv preprint forthcoming}
}
```

## License

MIT — see [LICENSE](LICENSE). Vendored third-party components (AutoLab,
RAGChecker, FIRE-Bench task data) keep their own licenses; see
[NOTICE](NOTICE).
