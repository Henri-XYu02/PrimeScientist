# Launchers

Four tmux launchers, one per (benchmark × method) pair. Each opens one tmux
window per task so runs proceed in parallel and survive a dropped SSH session.

| Script | Benchmark | Method | tmux session |
|---|---|---|---|
| `run_firebench.sh` | FIRE-Bench | MetaScientist tree search | `firebench` |
| `run_firebench_vanilla.sh` | FIRE-Bench | linear AutoResearch baseline | `firebench-vanilla` |
| `run_autolab.sh` | AutoLab | MetaScientist tree search | `autolab` |
| `run_autolab_vanilla.sh` | AutoLab | linear AutoResearch baseline | `autolab-vanilla` |

Run them from anywhere — each resolves the repo root from its own location:

```bash
bash scripts/run_firebench.sh
tmux attach -t firebench       # Ctrl-b w to list windows, Ctrl-b n/p to step
```

Each launcher activates the `metascientist` conda env via `conda run`. A window
stays open after its task finishes so you can read the final output; press
Enter to close it.

Re-running a launcher reuses an existing session and appends new windows.

---

## Anatomy of a launcher

Every script has the same three knobs near the top:

```bash
SESSION="firebench"                       # tmux session name
COMMON="--benchmark fire_bench ..."       # flags shared by every task
TASKS=(                                   # one tmux window per entry
    activation_control
    llm_value_consistency
    ...
)
```

**To run a subset**, comment out lines in `TASKS`. **To change budgets or
models**, edit `COMMON`. Task IDs are directory names under
`benchmarks/fire_bench/benchmark/papers/` or `benchmarks/autolab/tasks/`.

Two FIRE-Bench naming notes carried in the script comments: the task
`to_cot_or_not` maps to the directory `to_cot_or_not_to_cot`, and
`counterfactual_simulatability` and `neural_collapse_losses` expect a trimmed
`instruction.txt`.

---

## Smoke test

Before committing a full budget, shrink the run to a single evaluation:

```bash
# tree search: one node
python run_search.py --benchmark autolab --task gaussian_blur --budget 1

# baseline: one iteration
python run_vanilla.py --benchmark autolab --task gaussian_blur \
    --max_tokens 100000 --max_iters 1
```

Add `--skip_eval` to `run_search.py` to exercise the plumbing without paying
for evaluation.

---

## Flag reference — `run_search.py`

| Flag | Default | What |
|---|---|---|
| `--benchmark` | `fire_bench` | `fire_bench`, `autolab` |
| `--benchmark_path` | vendored dir | Override the benchmark repo location |
| `--task` | *required* | Task ID to search on |
| `--agent` | `codex` | Coding agent that executes each plan |
| `--model` | `gpt-5` | LLM backing the agent |
| `--reflector_agent` | `codex` | `codex` or `claude` |
| `--reflector_model` | `gpt-5` | LLM backing the reflector |
| `--reflector_timeout` | `600` | Reflector subprocess timeout (s) |
| `--budget` | `10` | Total agent runs (evaluations) |
| `--proposals` | `2` | Child plans proposed per reflection (`m`) |
| `--n_runs` | `1` | Runs per node; score = max across runs |
| `--alpha_max` | `10.0` | Cap on BAVT `α = 1/r_t` |
| `--max_depth` | `0` | Max tree depth; `0` = unlimited |
| `--prune_threshold` | `0.05` | Prune child if score < `parent_Q − δ`; `0` disables |
| `--max_tokens` | `0` | Token budget `B` (agent + reflector); `0` = unlimited |
| `--max_time_sec` | `0` | Wall-clock cap (s); `0` = unlimited |
| `--max_cost_usd` | `0` | Cost cap via `costs.tsv`; `0` = unlimited |
| `--output_dir` | `$FIRE_BENCH_SEARCH_TREE` or `./search_tree` | Search-tree root |
| `--force_init` | off | Re-initialise the root even if a tree exists |
| `--skip_eval` | off | Dry run: skip evaluation |

## Flag reference — `run_vanilla.py`

| Flag | Default | What |
|---|---|---|
| `--benchmark` | `autolab` | Benchmark name |
| `--benchmark_path` | vendored dir | Override the benchmark repo location |
| `--task` | *required* | Task ID to iterate on |
| `--agent` / `--model` | `codex` / `gpt-5` | Coding agent and its LLM |
| `--max_tokens` | *required* | Token budget — matches the search budget for a fair comparison |
| `--max_iters` | `100` | Safety cap on iterations |
| `--max_time_sec` | `0` | Wall-clock cap (s); `0` = unlimited |
| `--max_cost_usd` | `0` | Cost cap via `costs.tsv`; `0` = unlimited |
| `--output_dir` | `$FIRE_BENCH_VANILLA_RUNS` or `./vanilla_runs` | Run-output root |

Both entry points read `.env` through `python-dotenv`; see
[`../.env.example`](../.env.example) for the environment variables.

---

## Budget provenance

The `--max_tokens 1500000` in every launcher is not arbitrary; the derivation
is in each script's header comment. In short, measured on `codex`/`gpt-5`:

- **FIRE-Bench** — agent mean ≈ 60k, p90 ≈ 110k; reflector mean ≈ 75k. With
  `--n_runs 3`, one search round ≈ 255k tokens, so 1.5M ⇒ ~6 unique plans
  (18 agent calls), versus ~25 independent runs for the baseline.
- **AutoLab** — agent mean ≈ 66k, p90 ≈ 155k; reflector mean ≈ 75k. One round
  ≈ 140k, so 1.5M ⇒ ~10 rounds, versus ~20 baseline runs. Heavy tasks
  (`aes128_ctr`, `flash_attention`) get ~8 rounds.

Both methods run under the same cap, which is what makes the comparison fair.
