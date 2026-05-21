#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time as time_module
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MAIN_PATH = Path(__file__).parent

from search.tree import (
    TreeNode, init_tree, load_tree,
    puct_select_leaf,
    best_node, tree_summary, node_depth,
)
from search.reflector import init_task, reflect_and_propose
from benchmarks.registry import load_benchmark
from benchmarks.utils import append_cost_ledger, log_error

# ---------------------------------------------------------------------------
# TSV ledger
# ---------------------------------------------------------------------------

LEDGER_HEADER = "node\ttask\tscore\tran_at\n"


def _init_ledger(path: Path) -> None:
    if not path.exists():
        path.write_text(LEDGER_HEADER, encoding="utf-8")


def _append_ledger(path: Path, node: TreeNode, packet: dict, primary: float, pruned: bool = False) -> None:
    node_label = f"pruned:{node.path.name}" if pruned else node.path.name
    row = "\t".join([
        node_label,
        packet.get("task", ""),
        f"{primary:.4f}",
        packet.get("ran_at", ""),
    ])
    with open(path, "a", encoding="utf-8") as f:
        f.write(row + "\n")


def _read_total_tokens(cost_ledger: Path) -> int:
    """Sum input+output tokens across all rows in costs.tsv (agent + reflector)."""
    if not cost_ledger.exists():
        return 0
    total = 0
    try:
        lines = cost_ledger.read_text(encoding="utf-8").splitlines()[1:]  # skip header
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 6:
                total += int(parts[4] or 0) + int(parts[5] or 0)
    except Exception as e:
        log_error("_read_total_tokens", e, cost_ledger=cost_ledger)
    return total


def _compute_alpha(
    cost_ledger: Path,
    max_tokens: int,
    start_time: float,
    max_time_sec: int,
    alpha_max: float = 10.0,
) -> float:
    """Compute BAVT alpha = 1/r_t.

    r_t = min(remaining_token_ratio, remaining_time_ratio).
    Falls back to r_t=1 (alpha=1, pure exploration) when no budget limits are set.
    alpha is capped at alpha_max to avoid numerical issues near budget exhaustion.
    """
    ratios = []
    if max_tokens > 0:
        used = _read_total_tokens(cost_ledger)
        ratios.append(max(0.0, (max_tokens - used) / max_tokens))
    if max_time_sec > 0:
        elapsed = time_module.time() - start_time
        ratios.append(max(0.0, (max_time_sec - elapsed) / max_time_sec))
    if not ratios:
        return 1.0   # no budget → always explore proportionally to V
    r_t = min(ratios)
    if r_t <= 0.0:
        return alpha_max
    return min(1.0 / r_t, alpha_max)


def _bootstrap_proposal(child: TreeNode, parent: TreeNode) -> bool:
    """Validate and finalise a reflector-written proposal directory.

    The reflector is expected to have written skill.md and prior.json directly.
    This function checks the result and handles edge cases.

    Returns True if the proposal is valid (has skill.md), False if pruned.

    Edge cases:
    - skill.md missing                             → archive branch
    - prior.json missing / invalid / bad value     → default estimate=0.5
    """
    proposal_dir = child.path

    # ── skill.md — should have been written directly by the reflector ────────
    if not child.skill_path.exists():
        reason = "skill.md missing — reflector did not write it"
        (proposal_dir / "bootstrap_error.txt").write_text(reason, encoding="utf-8")
        archived = proposal_dir.parent / f".archived_{proposal_dir.name}"
        proposal_dir.rename(archived)
        print(f"  [bootstrap] Pruned {proposal_dir.name}: {reason}")
        return False

    # ── prior.json ───────────────────────────────────────────────────────────
    estimate = 0.5
    pf = proposal_dir / "prior.json"
    if pf.exists():
        try:
            pr       = json.loads(pf.read_text(encoding="utf-8"))
            estimate = float(pr.get("estimate", 0.5))
            estimate = max(0.0, min(1.0, estimate))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"  [prior] WARNING: bad prior.json for {proposal_dir.name} ({e}) — using 0.5")
    else:
        print(f"  [prior] prior.json missing for {proposal_dir.name} — using 0.5")
    child.set_prior(estimate)
    return True


MAX_REFLECT_RETRIES = 2


def _scan_raw_proposals(node: TreeNode, n_proposals: int) -> list[Path]:
    """Return unprocessed proposal dirs (have skill.md, no stats.json yet), capped at n_proposals."""
    children_dir = node.path / "children"
    if not children_dir.exists():
        return []
    return [
        d for d in sorted(children_dir.iterdir())
        if d.is_dir()
        and not d.name.startswith(".")         # exclude .archived_*, .pruned_*, etc.
        and not (d / "stats.json").exists()    # not yet bootstrapped
    ][:n_proposals]


def _prune_node(
    node: TreeNode,
    parent: TreeNode,
    primary: float,
    packet: dict,
    prune_threshold: float,
) -> None:
    """
    Mark a child as score-pruned:
      1. Write packet.json into the node dir (permanent record of the run).
      2. Rename the dir to .pruned_{name} so get_children() ignores it.
      3. Append a compact entry to parent/pruned_children.md (prior.json +
         score) so the reflector knows what approaches already failed.
    """
    node.write_packet(packet)

    node_name = node.path.name
    pruned_path = node.path.parent / f".pruned_{node_name}"
    # On re-expansion the reflector may pick a name that's already pruned
    # (`ls children/` doesn't show hidden .pruned_* dirs). Append a counter
    # so the rename doesn't collide with an existing pruned sibling.
    if pruned_path.exists():
        suffix = 2
        while (node.path.parent / f".pruned_{node_name}_{suffix}").exists():
            suffix += 1
        pruned_path = node.path.parent / f".pruned_{node_name}_{suffix}"
    node.path.rename(pruned_path)

    # ── pruned_children.md in parent ─────────────────────────────────────────
    log_path = parent.path / "pruned_children.md"
    if not log_path.exists():
        log_path.write_text(
            "# Pruned Children\n\n"
            "These child nodes were run but scored significantly below the parent.\n"
            "The reflector MUST NOT repeat these approaches.\n"
            "Read this before proposing new children.\n\n",
            encoding="utf-8",
        )

    entry_lines = [
        f"## {node_name}  score={primary:.4f}  parent_Q={parent.Q:.4f}"
        f"  (pruned: below parent by >{prune_threshold:.4f})",
    ]

    # Compact approach summary: prefer prior.json's structured fields, fall
    # back to a skill.md excerpt if prior.json is missing/unreadable (the
    # reflector occasionally skips writing it).
    approach_added = False
    prior_path = pruned_path / "prior.json"
    if prior_path.exists():
        try:
            pr = json.loads(prior_path.read_text(encoding="utf-8"))
            entry_lines.append(
                f"  hypothesis : {pr.get('hypothesis', '—')}\n"
                f"  rationale  : {pr.get('rationale',  '—')}\n"
                f"  changes    : {pr.get('changes',    '—')}\n"
                f"  risks      : {pr.get('risks',      '—')}"
            )
            approach_added = True
        except (json.JSONDecodeError, ValueError):
            pass
    if not approach_added:
        skill_path = pruned_path / "skill.md"
        if skill_path.exists():
            try:
                skill_text = skill_path.read_text(encoding="utf-8").strip()
                if len(skill_text) > 800:
                    skill_text = skill_text[:800] + " …[truncated]"
                indented = skill_text.replace("\n", "\n    ")
                entry_lines.append(f"  skill.md (no prior.json):\n    {indented}")
            except OSError:
                pass

    # Short verifier verdict (why it failed / how far off)
    rd = packet.get("reward_detail", {})
    vot = rd.get("verifier_output_tail", "")
    if vot:
        tail = vot[-300:].strip()
        entry_lines.append(f"\n  verifier tail:\n    {tail.replace(chr(10), chr(10) + '    ')}")

    entry_lines.append("")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(entry_lines) + "\n\n")

    print(
        f"  [prune] {node_name} → .pruned_{node_name}"
        f"  score={primary:.4f} < parent_Q={parent.Q:.4f} − {prune_threshold:.4f}"
    )


def _expand_with_retry(
    node: TreeNode, task: str, root: TreeNode,
    n_proposals: int, tree_dir: Path, insights_dir: Path,
    reflector_agent: str, reflector_model: str, reflector_timeout: int,
    is_reexpand: bool = False,
    inheritance_note: str = "",
    proposal_block: str = "",
    reflect_template: str = "",
) -> tuple[int, dict]:
    """Expand a node: bootstrap any existing proposals first, then ask the
    reflector only if needed. Retries up to MAX_REFLECT_RETRIES times if all
    proposals are pruned.

    is_reexpand=True means all previous children were score-pruned; the
    reflector receives the pruned_children.md context in its prompt.

    Returns (n_valid_proposals, reflector_cost_info).
    """
    reflector_cost: dict = {}

    for attempt in range(1, MAX_REFLECT_RETRIES + 1):
        if attempt > 1:
            print(f"  [reflect] Retry {attempt}/{MAX_REFLECT_RETRIES} …")

        raw = _scan_raw_proposals(node, n_proposals)

        if not raw:
            ok, cost = reflect_and_propose(
                task=task, root_node=root, expand_node=node,
                n_proposals=n_proposals, tree_dir=tree_dir,
                insights_dir=insights_dir, agent=reflector_agent,
                model=reflector_model, timeout=reflector_timeout,
                is_reexpand=is_reexpand,
                inheritance_note=inheritance_note,
                proposal_block=proposal_block,
                reflect_template=reflect_template,
            )
            if cost:
                reflector_cost = cost
            if not ok:
                print(f"  [reflect] Attempt {attempt}: reflector subprocess failed.")
                continue
            raw = _scan_raw_proposals(node, n_proposals)

        if not raw:
            print(f"  [reflect] Attempt {attempt}: reflector created no proposal dirs.")
            continue

        for proposal_dir in raw:
            child = TreeNode(proposal_dir, task)
            _bootstrap_proposal(child, node)

        n_ready = len(node.get_children())
        if n_ready > 0:
            print(f"  [reflect] Attempt {attempt}: {n_ready} proposals ready.")
            return n_ready, reflector_cost

        print(f"  [reflect] Attempt {attempt}: all {len(raw)} proposals pruned.")

    print(f"  [reflect] Gave up after {MAX_REFLECT_RETRIES} attempts — no valid proposals.")
    return 0, reflector_cost


def _move_artifacts(packet: dict, log_root: Path, node_path: Path, tree_dir: Path) -> None:
    """Move agent log and run sandbox into the node directory.

    The full agent log is written as ``.log.log`` (dot-prefixed, hidden from
    default ``ls``) so the reflector doesn't stumble into it and burn tokens
    re-reading the full 50–150k-token trajectory. The reflector reads
    ``log_brief.log`` instead, which it generates on demand.
    Paths in packet are updated to be relative to tree_dir (= reflector cwd).
    """
    for key, local_name in [("log_path", ".log.log"), ("repo_path", "sandbox")]:
        rel = packet.get(key)
        if not rel:
            continue
        src = log_root / rel
        dst = node_path / local_name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
        if dst.exists():
            packet[key] = str(dst.relative_to(tree_dir))
            print(f"  [artifacts] {key} → {packet[key]}")


# ---------------------------------------------------------------------------
# Node runner (used both sequentially and inside ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _run_node(
    node: TreeNode,
    task: str,
    agent: str,
    model: str,
    run_env: dict,
    benchmark,
    tree_dir: Path,
    skip_eval: bool,
    n_runs: int = 1,
) -> tuple[TreeNode, dict | None, float]:
    """
    Execute one node: setup → agent × n_runs → evaluate → move best artifacts.
    Returns (node, packet, primary_score).  packet=None signals failure.
    When n_runs > 1, the node's score is the best across all runs; costs from
    all runs are stored in packet["extra_run_costs"] for budget accounting.
    """
    if not benchmark.setup_node(task, node):
        return node, None, 0.0
    if skip_eval:
        return node, {"task": task, "score": {}, "ran_at": datetime.now().isoformat()}, 0.0

    best_packet:  dict | None = None
    best_primary: float       = -1.0
    extra_costs:  list        = []   # costs from non-best runs

    # Code inheritance: children start from the parent's best achieved sandbox
    # rather than the pristine baseline, so they build incrementally.
    parent_node = node.parent()
    env_override = None
    if parent_node is not None and parent_node.Q > 0:
        parent_sandbox = parent_node.path / "sandbox"
        if parent_sandbox.exists():
            env_override = parent_sandbox
            preamble_template = benchmark.agent_inheritance_preamble
            if preamble_template:
                benchmark.prepend_skill_context(
                    task, preamble_template.format(parent_q=parent_node.Q)
                )

    for i in range(max(1, n_runs)):
        if i > 0:
            print(f"  [n_runs] run {i+1}/{n_runs} for {node.path.name}")
        run_id = benchmark.run_agent(task, agent, model, run_env,
                                     env_override=env_override,
                                     node_dir=node.path)
        if run_id is None:
            continue
        packet  = benchmark.evaluate(task, run_id)
        primary = benchmark.primary_score(packet)

        if primary > best_primary:
            if best_packet is not None:
                extra_costs.append(best_packet.get("cost", {}))
            best_primary = primary
            best_packet  = packet
        else:
            extra_costs.append(packet.get("cost", {}))

    if best_packet is None:
        return node, None, 0.0

    if benchmark.log_root:
        _move_artifacts(best_packet, benchmark.log_root, node.path, tree_dir)
    if extra_costs:
        best_packet["extra_run_costs"] = extra_costs

    return node, best_packet, best_primary


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def run_search(
    benchmark_name: str,
    benchmark_path: str | None,
    task: str,
    agent: str,
    model: str,
    budget: int,
    n_proposals: int,
    output_dir: Path,
    reflector_agent: str = "codex",
    reflector_model: str = "o4-mini",
    force_init: bool = False,
    skip_eval: bool = False,
    reflector_timeout: int = 600,
    max_depth: int = 4,
    max_tokens: int = 0,
    max_time_sec: int = 0,
    alpha_max: float = 10.0,
    n_runs: int = 1,
    prune_threshold: float = 0.0,
) -> None:

    benchmark    = load_benchmark(benchmark_name, benchmark_path)
    tree_dir     = output_dir / task
    ledger_path  = tree_dir / "results.tsv"
    cost_ledger  = tree_dir / "costs.tsv"

    insights_dir = benchmark.insights_dir

    tree_dir.mkdir(parents=True, exist_ok=True)
    insights_dir.mkdir(parents=True, exist_ok=True)

    run_env = {**os.environ, "AGENT_ID": agent, "LLM_MODEL": model}

    # ── Load or initialise tree ──────────────────────────────────────────────
    root   = load_tree(tree_dir, task)
    is_new = root is None or force_init

    if is_new:
        if force_init and root is not None:
            # Purge run-specific artefacts so old costs/results don't bleed in.
            # Keep insights/ and instruction.md (task-level, not run-specific).
            for _name in ("root", "costs.tsv", "results.tsv",
                          "reflector_session.json", ".reflector_logs"):
                _target = tree_dir / _name
                if _target.is_dir():
                    shutil.rmtree(_target)
                elif _target.exists():
                    _target.unlink()
            print(f"\n[init] force_init: cleared old run data in {tree_dir}")

        _init_ledger(ledger_path)
        print(f"\n[init] New task '{task}' — initialising root node …")
        root = init_tree(tree_dir, task)
        ok = init_task(
            task=task,
            root_node=root,
            instruction_path=benchmark.get_instruction_path(task),
            insights_dir=insights_dir,
            agent=reflector_agent,
            model=reflector_model,
            tree_dir=tree_dir,
            timeout=reflector_timeout,
        )
        if not ok:
            print("[init] ERROR: reflector failed to initialise skill files.")
            return
        # Copy insight snapshot into root node (reflector writes it to insights_dir)
        live_insight = insights_dir / f"{task}_insight.md"
        if live_insight.exists():
            shutil.copy2(live_insight, root.insight_path)
        # Benchmark-specific post-init hook (e.g. fire_bench copies skill to live location)
        benchmark.post_init_node(task, root)
    else:
        _init_ledger(ledger_path)
        bq = best_node(root)[0]
        print(f"\n[resume] Resuming tree for '{task}'  best={bq:.3f}")

    # ── Main loop ────────────────────────────────────────────────────────────
    evals_done  = 0
    step        = 0
    start_time  = time_module.time()

    def _budget_status() -> str:
        parts = [f"evals={evals_done}/{budget}"]
        if max_tokens > 0:
            parts.append(f"tokens={_read_total_tokens(cost_ledger):,}/{max_tokens:,}")
        if max_time_sec > 0:
            parts.append(f"time={time_module.time()-start_time:.0f}s/{max_time_sec}s")
        return "  ".join(parts)

    def _over_budget() -> bool:
        if max_tokens > 0 and _read_total_tokens(cost_ledger) >= max_tokens:
            print(f"  [budget] Token limit reached: {_read_total_tokens(cost_ledger):,} >= {max_tokens:,}")
            return True
        if max_time_sec > 0 and (time_module.time() - start_time) >= max_time_sec:
            print(f"  [budget] Time limit reached: {time_module.time()-start_time:.0f}s >= {max_time_sec}s")
            return True
        return False

    while evals_done < budget:
        step += 1
        alpha = _compute_alpha(cost_ledger, max_tokens, start_time, max_time_sec, alpha_max)
        print(f"\n{'─'*60}")
        print(f"  Step {step}  |  {_budget_status()}  |  alpha={alpha:.2f}"
              f"  task={task}  benchmark={benchmark_name}")
        print(f"{'─'*60}")
        print(tree_summary(root))

        if _over_budget():
            break

        # ── BAVT: select next node ────────────────────────────────────────────
        node = puct_select_leaf(
            root, alpha=alpha, max_depth=max_depth,
            max_proposals=n_proposals,
        )
        print(f"\n  [bavt] → {node.path.name}  Q={node.Q:.3f}  P={node.P:.2f}"
              f"  depth={node_depth(node)}  alpha={alpha:.2f}")

        if not node.has_packet():
            # BAVT landed on unrun node → run it
            node_name = node.path.name  # capture before potential rename by _prune_node
            node, packet, primary = _run_node(node, task, agent, model, run_env,
                                              benchmark, tree_dir, skip_eval, n_runs)
            if packet is None:
                print(f"  [run] {node_name} failed — skipping.")
            else:
                parent   = node.parent()
                parent_q = parent.Q if parent else 0.0
                packet["delta_reward"] = round(primary - parent_q, 4)

                # ── Pruning check ────────────────────────────────────────────
                should_prune = (
                    parent is not None
                    and prune_threshold > 0.0
                    and parent_q > 0.0
                    and primary < parent_q - prune_threshold
                )

                if should_prune:
                    _prune_node(node, parent, primary, packet, prune_threshold)
                    # no backpropagate — pruned score must not drag parent Q down
                else:
                    node.write_packet(packet)
                    node.backpropagate(primary)

                _append_ledger(ledger_path, node, packet, primary, pruned=should_prune)
                append_cost_ledger(cost_ledger, packet.get("cost", {}),
                                   f"agent:{agent}", model, node_name)
                for ec in packet.get("extra_run_costs", []):
                    append_cost_ledger(cost_ledger, ec, f"agent:{agent}", model, node_name)
                evals_done += 1
                bq = best_node(root)[0]
                tag = "PRUNED" if should_prune else "result"
                print(f"  [{tag}] {node_name}  score={primary:.3f}"
                      f"  delta={packet['delta_reward']:+.3f}"
                      f"  best={bq:.3f}  {_budget_status()}")
                if _over_budget():
                    break
        else:
            # BAVT landed on a run node → expand with reflector.
            # This covers both first-time expansion AND re-expansion after all
            # children were pruned (get_children() excludes .pruned_* dirs).
            n_existing = len(node.get_children())
            n_pruned   = node.pruned_child_count()

            if n_existing >= n_proposals and n_pruned == 0:
                print(f"  [skip] {node.path.name} at cap ({n_existing}/{n_proposals}) "
                      f"and no deeper path available. Tree fully explored.")
                break

            n_can_add = n_proposals - n_existing
            if n_pruned > 0 and n_existing == 0:
                print(f"\n  [re-expand] {node.path.name}: all {n_pruned} children pruned"
                      f" — re-expanding with reflector (reflector reads pruned_children.md)")
            else:
                print(f"\n  [reflect] Expanding {node.path.name} "
                      f"({n_existing}/{n_proposals} children) — asking for up to {n_can_add} …")

            is_reexpand = (n_pruned > 0 and n_existing == 0)
            parent_has_sandbox = node.Q > 0 and (node.path / "sandbox").exists()
            inheritance_note = ""
            if parent_has_sandbox:
                note_template = benchmark.reflector_inheritance_note
                if note_template:
                    inheritance_note = note_template.format(parent_q=node.Q)
            n_ready, refl_cost = _expand_with_retry(
                node=node, task=task, root=root,
                n_proposals=n_can_add, tree_dir=tree_dir,
                insights_dir=insights_dir, reflector_agent=reflector_agent,
                reflector_model=reflector_model, reflector_timeout=reflector_timeout,
                is_reexpand=is_reexpand,
                inheritance_note=inheritance_note,
                proposal_block=benchmark.proposal_block,
                reflect_template=benchmark.reflect_template,
            )
            if refl_cost and node.has_packet():
                pkt = node.read_packet()
                pkt["reflector_cost"] = refl_cost
                node.write_packet(pkt)
            if n_ready == 0:
                print("  [reflect] Expansion failed after retries — stopping search.")
                break
            if _over_budget():
                break

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed  = time_module.time() - start_time
    total_tok = _read_total_tokens(cost_ledger)
    bq, best  = best_node(root)
    print(f"\n{'='*60}")
    print(f"Search done  benchmark={benchmark_name}  task={task}")
    print(f"  Best score:    {bq:.3f}  (node: {best.path.name})")
    print(f"  Evals:         {evals_done}/{budget}")
    print(f"  Total tokens:  {total_tok:,}" + (f"  (limit: {max_tokens:,})" if max_tokens else ""))
    print(f"  Elapsed:       {elapsed:.0f}s" + (f"  (limit: {max_time_sec}s)" if max_time_sec else ""))
    print(f"  Ledger:        {ledger_path}")
    print(f"\n{tree_summary(root)}")
    print(f"{'='*60}")

    # Search tree is the source of truth — no writes back to benchmark folder.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PUCT-guided self-improvement search for any benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark",      default="fire_bench",
                        help="Benchmark name (fire_bench, autolab, …)")
    parser.add_argument("--benchmark_path", default=None,
                        help="Path to benchmark repo (required for autolab)")
    parser.add_argument("--task",           required=True,
                        help="Task ID to search on")
    parser.add_argument("--agent",          default="codex",
                        help="Agent to run tasks (codex, terminus-2, …)")
    parser.add_argument("--model",          default="gpt-5",
                        help="LLM model for the agent")
    parser.add_argument("--budget",         type=int, default=10,
                        help="Total agent runs (evaluations)")
    parser.add_argument("--proposals",      type=int, default=2,
                        help="Skill variants proposed per reflection")
    parser.add_argument("--output_dir",     default=os.environ.get("FIRE_BENCH_SEARCH_TREE", "./search_tree"),
                        help="Root dir for search trees (one subdir per task)")
    parser.add_argument("--alpha_max",       type=float, default=10.0,
                        help="Cap on BAVT alpha=1/r_t; prevents extreme exploitation at near-zero budget")
    parser.add_argument("--max_depth",      type=int,   default=0,
                        help="Max tree depth; 0 = unlimited")
    parser.add_argument("--reflector_agent", default="codex",
                        choices=["claude", "codex"],
                        help="Coding agent used as reflector")
    parser.add_argument("--reflector_model", default="gpt-5",
                        help="Model for the reflector agent (e.g. o4-mini, claude-opus-4-6)")
    parser.add_argument("--reflector_timeout", type=int, default=600,
                        help="Reflector subprocess timeout in seconds")
    parser.add_argument("--n_runs",         type=int,   default=1,
                        help="Number of agent runs per node; node score = max across runs (default 1)")
    parser.add_argument("--max_tokens",     type=int,   default=0,
                        help="Stop when total input+output tokens (agent+reflector) exceed this; 0=unlimited")
    parser.add_argument("--max_time_sec",   type=int,   default=0,
                        help="Stop after this many seconds of wall-clock time; 0=unlimited")
    parser.add_argument("--prune_threshold", type=float, default=0,
                        help="Prune a child if its score < parent_Q - threshold; 0=disabled")
    parser.add_argument("--force_init",     action="store_true",
                        help="Re-initialise root even if tree exists")
    parser.add_argument("--skip_eval",      action="store_true",
                        help="Skip evaluation (dry run)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"\n{'='*60}")
    print(f"Self-improvement search")
    print(f"  benchmark : {args.benchmark}")
    print(f"  task      : {args.task}")
    print(f"  agent     : {args.agent}/{args.model}")
    print(f"  budget    : {args.budget}  proposals: {args.proposals}")
    print(f"  reflector : {args.reflector_agent}/{args.reflector_model}")
    print(f"  alpha_max : {args.alpha_max}  max_depth: {args.max_depth}  n_runs: {args.n_runs}"
          + (f"  prune_threshold: {args.prune_threshold}" if args.prune_threshold > 0 else ""))
    print((f"  max_tokens: {args.max_tokens:,}" if args.max_tokens else "")
          + (f"  max_time: {args.max_time_sec}s" if args.max_time_sec else ""))
    print(f"  output    : {output_dir / args.task}")
    print(f"{'='*60}")

    run_search(
        benchmark_name=args.benchmark,
        benchmark_path=args.benchmark_path,
        task=args.task,
        agent=args.agent,
        model=args.model,
        budget=args.budget,
        n_proposals=args.proposals,
        output_dir=output_dir,
        reflector_agent=args.reflector_agent,
        reflector_model=args.reflector_model,
        force_init=args.force_init,
        skip_eval=args.skip_eval,
        reflector_timeout=args.reflector_timeout,
        max_depth=args.max_depth,
        max_tokens=args.max_tokens,
        max_time_sec=args.max_time_sec,
        alpha_max=args.alpha_max,
        n_runs=args.n_runs,
        prune_threshold=args.prune_threshold,
    )


if __name__ == "__main__":
    main()
