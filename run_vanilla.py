#!/usr/bin/env python3
"""
run_vanilla.py — Vanilla AutoResearch baseline (linear edit-run-keep-or-revert).

Reproduces the loop from https://github.com/karpathy/autoresearch , which is
the standard non-tree-search baseline referenced in docs/paper.tex:

  while budget not exhausted:
      copy best_state to a fresh iter_sandbox
      inject a HISTORY.md summarising previous iterations into the sandbox
      run the coding agent on the sandbox (agent edits code in-place)
      evaluate
      if score improves: commit — replace best_state with iter_sandbox
      else:               revert — discard iter_sandbox, best_state unchanged

Compared with our tree-search approach (run_search.py) this baseline:
  - does no tree branching (purely linear history)
  - has no reflector subprocess (agent both proposes and executes)
  - still respects the same --max_tokens budget cap, for a fair comparison

Directory layout per task:
  <output_dir>/{task}/
      best_sandbox/              persistent "best" code state (git-commit analogue)
      iter_000/
          sandbox/               agent's edit of best_sandbox for this iteration
          packet.json
          log.log
      iter_001/
          ...
      results.tsv                iter, score, delta, action, tokens_cum
      costs.tsv                  same format as run_search.py's costs.tsv

Supports both autolab (code editing on a persistent codebase) and fire_bench
(research replication; iteration 0 builds from data+utils, later iterations
seed from best_sandbox via the benchmark's inheritance flow).
"""

import argparse
import json
import os
import shutil
import time as time_module
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MAIN_PATH = Path(__file__).parent
DATA_PATH = Path("/home/xinle/FIRE-Bench")

from benchmarks.registry import load_benchmark
from benchmarks.utils import append_cost_ledger, log_error


LEDGER_HEADER = "iter\tscore\tdelta\taction\ttokens_cum\tran_at\n"


# ---------------------------------------------------------------------------
# Token budget tracking
# ---------------------------------------------------------------------------

def _read_total_tokens(cost_ledger: Path) -> int:
    if not cost_ledger.exists():
        return 0
    total = 0
    try:
        lines = cost_ledger.read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 6:
                total += int(parts[4] or 0) + int(parts[5] or 0)
    except Exception as e:
        log_error("_read_total_tokens", e, cost_ledger=cost_ledger)
    return total


# ---------------------------------------------------------------------------
# History (shown to agent as a "skill" document)
# ---------------------------------------------------------------------------

HISTORY_HEADER = """# Prior Experiment Attempts

You are in a linear experiment loop. Each previous attempt below shows the
change that was made and the resulting score. An attempt marked **kept** means
that iteration's code IS now the code you see in this workspace. An attempt
marked **reverted** means its changes were discarded and the codebase was
restored to the best prior state.

Use this history to avoid repeating failed approaches and to build on what
worked. Then make ONE new change that you expect to improve the score.

"""


def _format_history(results_tsv: Path, packets: list[dict], max_entries: int = 10) -> str:
    """Render past iterations as a markdown log shown to the agent."""
    if not packets:
        return HISTORY_HEADER + "_(no prior attempts — this is iteration 0)_\n"
    lines = [HISTORY_HEADER]
    # Keep the last max_entries to bound prompt size
    shown = packets[-max_entries:]
    if len(packets) > max_entries:
        lines.append(f"_(showing last {max_entries} of {len(packets)} attempts)_\n")
    for entry in shown:
        i      = entry["iter"]
        score  = entry["score"]
        delta  = entry["delta"]
        action = entry["action"]
        summary = entry.get("summary", "").strip()[:400] or "(no summary)"
        lines.append(f"## Iteration {i}  —  score={score:.4f}  Δ={delta:+.4f}  **{action}**\n")
        lines.append(summary + "\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sandbox commit/revert
# ---------------------------------------------------------------------------

# Excluded from best_sandbox commits:
#   _reward, target  — autolab build dirs (root-owned inside Docker)
#   .agents, .env    — per-run skill stage and API keys (regenerated each run)
#   CLAUDE.md        — agent meta-file
#   __pycache__      — Python bytecode
COMMIT_IGNORE = shutil.ignore_patterns("_reward", ".agents", ".env", "CLAUDE.md", "__pycache__", "target")


def _commit(iter_sandbox: Path, best_sandbox: Path) -> None:
    """Replace best_sandbox with iter_sandbox contents (kept changes)."""
    if best_sandbox.exists():
        shutil.rmtree(best_sandbox)
    shutil.copytree(str(iter_sandbox), str(best_sandbox), ignore=COMMIT_IGNORE)


# ---------------------------------------------------------------------------
# Artefact handling
# ---------------------------------------------------------------------------

def _move_artifacts(packet: dict, log_root: Path, iter_dir: Path, task_dir: Path) -> None:
    """Move agent log and sandbox into iter_dir, update packet paths."""
    for key, local_name in [("log_path", "log.log"), ("repo_path", "sandbox")]:
        rel = packet.get(key)
        if not rel:
            continue
        src = log_root / rel
        dst = iter_dir / local_name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
        if dst.exists():
            packet[key] = str(dst.relative_to(task_dir))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_vanilla(
    benchmark_name: str,
    benchmark_path: str | None,
    task: str,
    agent: str,
    model: str,
    output_dir: Path,
    max_tokens: int,
    max_iters: int,
    max_time_sec: int = 0,
) -> None:

    benchmark    = load_benchmark(benchmark_name, benchmark_path)
    task_dir     = output_dir / task
    best_sandbox = task_dir / "best_sandbox"
    cost_ledger  = task_dir / "costs.tsv"
    results_tsv  = task_dir / "results.tsv"

    task_dir.mkdir(parents=True, exist_ok=True)
    if not results_tsv.exists():
        results_tsv.write_text(LEDGER_HEADER, encoding="utf-8")

    run_env = {**os.environ, "AGENT_ID": agent, "LLM_MODEL": model}

    # ── Replay any prior iterations from disk so we can resume ──────────────
    packets: list[dict] = []
    best_score = 0.0
    start_iter = 0
    for i in range(10_000):
        pd = task_dir / f"iter_{i:03d}" / "packet.json"
        if not pd.exists():
            start_iter = i
            break
        try:
            p = json.loads(pd.read_text(encoding="utf-8"))
            sc = benchmark.primary_score(p)
            packets.append({
                "iter":    i,
                "score":   sc,
                "delta":   p.get("delta", 0.0),
                "action":  p.get("action", "?"),
                "summary": p.get("agent_summary", ""),
            })
            if p.get("action") == "kept":
                best_score = max(best_score, sc)
        except Exception:
            start_iter = i
            break

    if start_iter > 0:
        print(f"[resume] Found {start_iter} prior iterations  best_score={best_score:.3f}")

    # ── Main loop ────────────────────────────────────────────────────────────
    start_time  = time_module.time()

    def _budget_status() -> str:
        parts = [f"iter={len(packets)}/{max_iters}"]
        parts.append(f"tokens={_read_total_tokens(cost_ledger):,}/{max_tokens:,}")
        if max_time_sec > 0:
            parts.append(f"time={time_module.time()-start_time:.0f}s/{max_time_sec}s")
        return "  ".join(parts)

    def _over_budget() -> bool:
        used = _read_total_tokens(cost_ledger)
        if max_tokens > 0 and used >= max_tokens:
            print(f"  [budget] Token limit reached: {used:,} >= {max_tokens:,}")
            return True
        if max_time_sec > 0 and (time_module.time() - start_time) >= max_time_sec:
            print(f"  [budget] Time limit reached")
            return True
        return False

    i = start_iter
    while i < max_iters:
        print(f"\n{'─'*60}")
        print(f"  Iteration {i}  |  {_budget_status()}  |  task={task}")
        print(f"{'─'*60}")

        if _over_budget():
            break

        iter_dir = task_dir / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # ── Stage history as the injected skill ─────────────────────────────
        history_md = _format_history(results_tsv, packets)
        benchmark.stage_skill(task, history_md)
        (iter_dir / "history_shown.md").write_text(history_md, encoding="utf-8")

        # ── Run agent with sandbox seeded from best_sandbox ─────────────────
        # iter 0 (or any time we have no meaningful best): env_override=None →
        # autolab falls back to its pristine environment/, fire_bench falls
        # back to run.py (fresh data + utils via the agent script).
        env_override = best_sandbox if (best_sandbox.exists() and best_score > 0) else None
        run_id = benchmark.run_agent(task, agent, model, run_env, env_override=env_override)
        if run_id is None:
            print(f"  [run] iter {i}: agent run failed — skipping.")
            i += 1
            continue

        packet  = benchmark.evaluate(task, run_id)
        score   = benchmark.primary_score(packet)
        delta   = score - best_score

        action = "kept" if score > best_score else "reverted"
        packet["delta"]         = round(delta, 4)
        packet["action"]        = action
        packet["iter"]          = i
        packet["best_so_far"]   = max(best_score, score)

        # ── Move artefacts into iter_dir ─────────────────────────────────────
        if benchmark.log_root:
            try:
                _move_artifacts(packet, benchmark.log_root, iter_dir, task_dir)
            except Exception as e:
                log_error("_move_artifacts", e, task=task, iter=i)

        try:
            (iter_dir / "packet.json").write_text(json.dumps(packet, indent=2), encoding="utf-8")
        except Exception as e:
            log_error("packet_write", e, task=task, iter=i)
            print(f"  [warn] iter {i}: packet.json write failed (disk full?) — result NOT saved")

        # ── Ledgers ──────────────────────────────────────────────────────────
        try:
            append_cost_ledger(cost_ledger, packet.get("cost", {}),
                               f"agent:{agent}", model, f"iter_{i:03d}")
        except Exception as e:
            log_error("cost_ledger", e, task=task, iter=i)
        used_cum = _read_total_tokens(cost_ledger)
        try:
            with open(results_tsv, "a", encoding="utf-8") as f:
                f.write("\t".join([
                    str(i),
                    f"{score:.4f}",
                    f"{delta:+.4f}",
                    action,
                    str(used_cum),
                    packet.get("ran_at", ""),
                ]) + "\n")
        except Exception as e:
            log_error("results_tsv", e, task=task, iter=i)

        # ── Commit / revert ──────────────────────────────────────────────────
        if action == "kept":
            iter_sandbox = iter_dir / "sandbox"
            if iter_sandbox.exists():
                try:
                    _commit(iter_sandbox, best_sandbox)
                    print(f"  [commit] score {score:.4f} > prev best {best_score:.4f} → best_sandbox updated")
                except Exception as e:
                    log_error("_commit", e, task=task, iter=i)
                    print(f"  [warn] _commit failed ({e}) — best_sandbox NOT updated")
            best_score = score
        else:
            print(f"  [revert] score {score:.4f} ≤ best {best_score:.4f} → best_sandbox unchanged")

        packets.append({
            "iter":    i,
            "score":   score,
            "delta":   delta,
            "action":  action,
            "summary": packet.get("agent_conclusion", "") or packet.get("agent_summary", ""),
        })

        print(f"  [result] iter={i}  score={score:.4f}  Δ={delta:+.4f}  {action}  best={best_score:.4f}  {_budget_status()}")

        if _over_budget():
            break
        i += 1

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed  = time_module.time() - start_time
    total_tok = _read_total_tokens(cost_ledger)
    print(f"\n{'='*60}")
    print(f"Vanilla baseline done  task={task}")
    print(f"  Best score:    {best_score:.4f}")
    print(f"  Iterations:    {len(packets)}")
    print(f"  Total tokens:  {total_tok:,}  (limit: {max_tokens:,})")
    print(f"  Elapsed:       {elapsed:.0f}s")
    print(f"  Results:       {results_tsv}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vanilla AutoResearch baseline (linear edit-run-keep-or-revert).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark",      default="autolab",
                        help="Benchmark name (currently only autolab supported)")
    parser.add_argument("--benchmark_path", default=None,
                        help="Path to benchmark repo (required for autolab)")
    parser.add_argument("--task",           required=True,
                        help="Task ID to iterate on")
    parser.add_argument("--agent",          default="codex",
                        help="Coding agent (codex, claude, ...)")
    parser.add_argument("--model",          default="gpt-5",
                        help="LLM model for the agent")
    parser.add_argument("--output_dir",     default="/data/xinle/FIRE-Bench/vanilla_runs",
                        help="Root dir for vanilla runs (one subdir per task)")
    parser.add_argument("--max_tokens",     type=int, required=True,
                        help="Stop when total input+output tokens exceed this")
    parser.add_argument("--max_iters",      type=int, default=100,
                        help="Safety cap on number of iterations")
    parser.add_argument("--max_time_sec",   type=int, default=0,
                        help="Stop after this many seconds of wall-clock time; 0=unlimited")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"\n{'='*60}")
    print(f"Vanilla AutoResearch baseline")
    print(f"  benchmark : {args.benchmark}")
    print(f"  task      : {args.task}")
    print(f"  agent     : {args.agent}/{args.model}")
    print(f"  max_tokens: {args.max_tokens:,}")
    print(f"  max_iters : {args.max_iters}")
    print(f"  output    : {output_dir / args.task}")
    print(f"{'='*60}")

    run_vanilla(
        benchmark_name=args.benchmark,
        benchmark_path=args.benchmark_path,
        task=args.task,
        agent=args.agent,
        model=args.model,
        output_dir=output_dir,
        max_tokens=args.max_tokens,
        max_iters=args.max_iters,
        max_time_sec=args.max_time_sec,
    )


if __name__ == "__main__":
    main()
