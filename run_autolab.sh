#!/usr/bin/env bash
# run_autolab.sh — AutoLab suite for FIRE-Bench evaluation.
#
# Task set from benchmark_set.txt (autolab section).
#
# Per-task token budget: 1.5M
#   Derived from 20 prior agent runs + reflector runs on codex/gpt-5:
#     agent   mean≈66k  p90≈155k  max≈195k (aes128_ctr heaviest)
#     reflect mean≈75k  max≈95k
#   One tree-search "round" = 1 agent + amortised reflector ≈ 140k.
#   1.5M ⇒ ~10 rounds for our method  /  ~20 independent runs for the
#   vanilla autoresearch baseline running under the same cap.
#   For heavy tasks (aes128_ctr, flash_attention) expect ~8 rounds.
#
# Each task runs in its own tmux window inside the session "autolab".
# Usage:
#   ./run_autolab.sh               # launch all tasks
#   tmux attach -t autolab         # monitor progress

set -e

SESSION="autolab"
COMMON="--benchmark autolab --agent codex --model gpt-5 --reflector_agent codex --reflector_model gpt-5 --proposals 2 --budget 15 --max_tokens 1500000 --force_init"

TASKS=(
    gaussian_blur
    hash_join
    concurrent_kv_wal
    flash_attention
    # fft_rust
    # vliw_scheduler
    # smallest_game_player
)

WORKDIR="$(cd "$(dirname "$0")" && pwd)"

# Create session (detached) or reuse existing one.
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "main"
fi

for task in "${TASKS[@]}"; do
    cmd="conda run --no-capture-output -n firebench python run_search.py $COMMON --task ${task}"
    echo ">>> Launching autolab/${task} in tmux window '${task}'"
    tmux new-window -t "$SESSION" -n "$task" \
        -c "$WORKDIR" \
        "bash -c '${cmd}; echo; echo \"=== ${task} done (exit \$?) ===\"; read -p \"Press Enter to close...\"'"
done

echo ""
echo "All tasks launched. Attach with:  tmux attach -t ${SESSION}"
echo "Switch windows:  Ctrl-b w   (interactive list)"
echo "                 Ctrl-b n / Ctrl-b p   (next / prev)"
