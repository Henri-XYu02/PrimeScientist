#!/usr/bin/env bash
# run_autolab_vanilla.sh — Vanilla AutoResearch baseline on the AutoLab suite.
#
# Mirrors https://github.com/karpathy/autoresearch: a linear
# edit-run-keep-or-revert loop with no tree branching and no reflector.
# Runs under the *same* per-task token budget as run_autolab.sh (1.5M),
# so the two can be compared fairly: our method spends budget on agent +
# reflector + tree exploration, vanilla spends it all on agent iterations.
#
# Each task runs in its own tmux window inside the session "autolab-vanilla".
# Usage:
#   ./run_autolab_vanilla.sh               # launch all tasks
#   tmux attach -t autolab-vanilla         # monitor progress

set -e

SESSION="autolab-vanilla"
COMMON="--benchmark autolab --agent codex --model gpt-5 --max_tokens 1500000 --max_iters 25"

TASKS=(
    gaussian_blur
    hash_join
    concurrent_kv_wal
    flash_attention
    fft_rust
    vliw_scheduler
    smallest_game_player
)

WORKDIR="$(cd "$(dirname "$0")" && pwd)"

# Create session (detached) or reuse existing one.
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "main"
fi

for task in "${TASKS[@]}"; do
    cmd="conda run --no-capture-output -n treescientist python run_vanilla.py $COMMON --task ${task}"
    echo ">>> Launching autolab-vanilla/${task} in tmux window '${task}'"
    tmux new-window -t "$SESSION" -n "$task" \
        -c "$WORKDIR" \
        "bash -c '${cmd}; echo; echo \"=== ${task} done (exit \$?) ===\"; read -p \"Press Enter to close...\"'"
done

echo ""
echo "All tasks launched. Attach with:  tmux attach -t ${SESSION}"
echo "Switch windows:  Ctrl-b w   (interactive list)"
echo "                 Ctrl-b n / Ctrl-b p   (next / prev)"
