#!/usr/bin/env bash
# run_firebench.sh — FIRE-Bench paper-replication suite.
#
# Task set from benchmark_set.txt (firebench section).
# `to_cot_or_not` in the set maps to the directory `to_cot_or_not_to_cot`.
# `counterfactual_simulatability` and `neural_collapse_losses` both need
# trimmed instruction.txt (note in benchmark_set.txt).
#
# Per-task token budget: 1.5M
#   Derived from 19 prior agent runs on codex/gpt-5:
#     agent   mean≈60k  p90≈110k  max≈111k (cot_in_planning heaviest)
#     reflect mean≈75k  max≈95k
#   With --n_runs 3 (each plan evaluated 3x, score = max), one tree-search
#   "round" = 3 agent + amortised reflector ≈ 255k tokens.
#   1.5M ⇒ ~6 unique plans (18 agent calls) for our method  /  ~25
#   independent single runs for the vanilla autoresearch baseline
#   running under the same cap.
#
# Each task runs in its own tmux window inside the session "firebench".
# Usage:
#   ./run_firebench.sh          # launch all tasks
#   tmux attach -t firebench           # monitor progress

set -e
export FIREBENCH_USE_DOCKER=1

SESSION="firebench"
COMMON="--benchmark fire_bench --agent codex --model gpt-5 --reflector_agent codex --reflector_model gpt-5 --proposals 3 --n_runs 1 --budget 15 --max_tokens 1500000 --prune_threshold 0.05"

TASKS=(
    activation_control
    llm_value_consistency
    seca_hallucination
    to_cot_or_not_to_cot
    questbench
    learning_order_agreement
    grokking_or_not
    max_suppression
    counterfactual_simulatability
    neural_collapse_losses
)

WORKDIR="$(cd "$(dirname "$0")" && pwd)"

# Create session (detached) or reuse existing one.
if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "main"
fi

for task in "${TASKS[@]}"; do
    cmd="conda run --no-capture-output -n treescientist python run_search.py $COMMON --task ${task}"
    echo ">>> Launching fire_bench/${task} in tmux window '${task}'"
    tmux new-window -t "=$SESSION" -n "$task" \
        -c "$WORKDIR" \
        "bash -c '${cmd}; echo; echo \"=== ${task} done (exit \$?) ===\"; read -p \"Press Enter to close...\"'"
done

echo ""
echo "All tasks launched. Attach with:  tmux attach -t ${SESSION}"
echo "Switch windows:  Ctrl-b w   (interactive list)"
echo "                 Ctrl-b n / Ctrl-b p   (next / prev)"
