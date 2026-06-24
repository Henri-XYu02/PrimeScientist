#!/usr/bin/env bash
# run_firebench_vanilla.sh — Vanilla AutoResearch baseline on the FireBench suite.
#
# Mirrors run_autolab_vanilla.sh: linear edit-run-keep-or-revert loop with no
# tree branching and no reflector. Runs under the same per-task token budget
# as the tree-search method (1.5M), so the two can be compared fairly.
#
# fire_bench specifics (handled inside run_vanilla.py):
#   - iter 0 has no parent sandbox; fire_bench's run.py builds the workspace
#     fresh from data + utils.
#   - iter 1+ seeds from best_sandbox via run_inherit.py.
#   - Per-iteration history is staged as the agent's research-agent-skill so
#     it sees what was already tried.
#
# Each task runs in its own tmux window inside the session "firebench-vanilla".
# Usage:
#   ./run_firebench_vanilla.sh             # launch all tasks
#   tmux attach -t firebench-vanilla       # monitor progress

set -e
export FIREBENCH_USE_DOCKER=1

SESSION="firebench-vanilla"
COMMON="--benchmark fire_bench --agent codex --model gpt-5 --max_tokens 1500000 --max_iters 15"

TASKS=(
    activation_control # (done)
    llm_value_consistency
    seca_hallucination # (done)
    to_cot_or_not_to_cot
    questbench
    learning_order_agreement
    max_suppression
    counterfactual_simulatability
    neural_collapse_losses
    grokking_or_not
)

WORKDIR="$(cd "$(dirname "$0")" && pwd)"

# Create session (detached) or reuse existing one.
if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "main"
fi

for task in "${TASKS[@]}"; do
    cmd="conda run --no-capture-output -n metascientist python run_vanilla.py $COMMON --task ${task}"
    echo ">>> Launching firebench-vanilla/${task} in tmux window '${task}'"
    tmux new-window -t "=$SESSION" -n "$task" \
        -c "$WORKDIR" \
        "bash -c '${cmd}; echo; echo \"=== ${task} done (exit \$?) ===\"; read -p \"Press Enter to close...\"'"
done

echo ""
echo "All tasks launched. Attach with:  tmux attach -t ${SESSION}"
echo "Switch windows:  Ctrl-b w   (interactive list)"
echo "                 Ctrl-b n / Ctrl-b p   (next / prev)"
