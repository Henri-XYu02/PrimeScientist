"""
benchmarks/fire_bench/__init__.py — FIRE-Bench benchmark implementation.

Agent runs code to replicate a research paper's key finding.
Evaluator: RAGChecker (precision / recall / F1 over the extracted conclusion).
Primary score: F1.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmarks.base import Benchmark
from benchmarks.utils import parse_run_cost, print_cost
from search.reflector import CODEACT_PROPOSAL_BLOCK, FIRE_BENCH_REFLECT_PROMPT
from search.tree import TreeNode

FIRE_BENCH_DIR = Path(__file__).parent            # benchmarks/fire_bench/
MAIN_PATH      = FIRE_BENCH_DIR.parent.parent     # repo root
DATA_PATH = Path(os.environ.get("FIRE_BENCH_DATA") or MAIN_PATH / "fire_bench_data")


class FireBench(Benchmark):

    def __init__(self, benchmark_path: Path | None = None):
        self._root      = benchmark_path or MAIN_PATH
        self._data_path = DATA_PATH
        self._log_dir   = DATA_PATH / "log"
        self._runs_dir  = DATA_PATH / "runs"
        self._results_dir = DATA_PATH / "results"

    @property
    def name(self) -> str:
        return "fire_bench"

    @property
    def insights_dir(self) -> Path:
        return FIRE_BENCH_DIR / "benchmark" / "insights"

    # ── Instruction ──────────────────────────────────────────────────────────

    def get_instruction_path(self, task: str) -> Path:
        return FIRE_BENCH_DIR / "benchmark" / "papers" / task / "instruction" / "instruction.txt"

    # ── Node setup ───────────────────────────────────────────────────────────

    def prepend_skill_context(self, task: str, preamble: str) -> None:
        skill_dst = FIRE_BENCH_DIR / "benchmark" / "papers" / task / "skill" / "skill.md"
        if skill_dst.exists():
            skill_dst.write_text(preamble + skill_dst.read_text(encoding="utf-8"), encoding="utf-8")

    def stage_skill(self, task: str, content: str) -> None:
        skill_dst = FIRE_BENCH_DIR / "benchmark" / "papers" / task / "skill" / "skill.md"
        skill_dst.parent.mkdir(parents=True, exist_ok=True)
        skill_dst.write_text(content, encoding="utf-8")

    @property
    def proposal_block(self) -> str:
        return CODEACT_PROPOSAL_BLOCK

    @property
    def reflect_template(self) -> str:
        return FIRE_BENCH_REFLECT_PROMPT

    @property
    def agent_inheritance_preamble(self) -> str:
        return (
            "> **INHERITED WORKSPACE** (score={parent_q:.3f}): The sandbox is "
            "seeded from the parent node's best research scripts. The skill "
            "below is the COMPLETE updated experimental plan; see "
            "`./INHERITED_FROM_PARENT/changes.md` for a short prose summary "
            "of what differs from the parent's plan and "
            "`./INHERITED_FROM_PARENT/diff.py` for the script that produced "
            "it. Update your scripts wherever the plan differs; you do not "
            "need to re-do steps the parent completed correctly.\n\n"
        )

    @property
    def reflector_inheritance_note(self) -> str:
        return (
            "\n\nNOTE — code inheritance active: child agents will be seeded "
            "from this node's `sandbox/` directory (parent Q={parent_q:.3f}). "
            "The sandbox already contains the parent's research scripts, but "
            "research-replication tasks need a COMPLETE experimental plan to "
            "produce a correct conclusion. Each child's skill.md must be a "
            "FULL self-contained plan (dataset, model, hyperparameters, "
            "evaluation metrics, conclusion structure) — clearly mark the "
            "section(s) that differ from the parent. Do NOT write a delta-only "
            "skill; the agent needs the full plan to know what to run.\n"
        )

    def setup_node(self, task: str, node: TreeNode) -> bool:
        """Copy skill.md snapshot to live benchmark location so the agent can read it."""
        skill_dst = FIRE_BENCH_DIR / "benchmark" / "papers" / task / "skill" / "skill.md"
        skill_dst.parent.mkdir(parents=True, exist_ok=True)

        if not node.skill_path.exists():
            print(f"  [setup] ERROR: {node.skill_path} missing.")
            return False

        shutil.copy2(node.skill_path, skill_dst)
        return True

    # ── Agent run ─────────────────────────────────────────────────────────────

    def run_agent(self, task: str, agent: str, model: str, run_env: dict,
                  env_override=None, node_dir=None) -> str | None:
        """Run agent once via run_agent.py. Returns timestamp_id or None.

        When env_override points to a parent node's sandbox, run_agent.py
        dispatches to the agent's run_inherit.py which seeds the workspace
        from that sandbox instead of starting from pristine data + utils.
        node_dir is the child node's tree dir — passed so the inherit
        script can copy diff.py and changes.md into the agent's sandbox.
        """
        since = time.time()
        if env_override is not None:
            run_env = {**run_env, "FIRE_BENCH_SEED_REPO": str(env_override)}
        if node_dir is not None:
            run_env = {**run_env, "FIRE_BENCH_NODE_DIR": str(node_dir)}
        cmd = [
            sys.executable, str(FIRE_BENCH_DIR / "run_agent.py"),
            "--AGENT_ID", agent,
            "--TASK_ID", task,
            "--LLM_MODEL", model,
            "--RUN_TIMES", "1",
            "--MAX_PARALLEL", "1",
        ]
        print(f"  [run] {agent}/{model} on {task}")
        result = subprocess.run(cmd, env=run_env, check=False)
        if result.returncode != 0:
            print(f"  [run] WARNING: run_agent.py exited {result.returncode}")

        records = self._find_new_logs([task], agent, model, since)
        return records[0]["timestamp_id"] if records else None

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, task: str, run_id: str) -> dict:
        """Run RAGChecker and return a packet dict."""
        since_log  = time.time() - 5   # small buffer

        # Find the log record for this run_id
        records = self._find_new_logs([task], "*", "*", since_log - 3600, run_id=run_id)
        if not records:
            print(f"  [eval] WARNING: log not found for run_id={run_id}")
            records = [{"task": task, "log_path": "", "repo_path": None,
                        "timestamp_id": run_id, "score": {}}]

        rec = records[0]

        # RAGChecker evaluation
        self._results_dir.mkdir(parents=True, exist_ok=True)
        before_eval = set(self._results_dir.glob("eval_results_*.json"))
        before_ea   = set(self._results_dir.glob("error_analysis_results_*.json"))

        eval_script = FIRE_BENCH_DIR / "eval" / "RAGChecker" / "eval.py"
        # Derive agent+model from log_path if available
        parts = Path(rec["log_path"]).parts if rec["log_path"] else []
        agent = parts[1] if len(parts) > 1 else "all"
        model = parts[2] if len(parts) > 2 else "all"

        cmd = [
            sys.executable, str(eval_script),
            "--agents", agent,
            "--models", model,
            "--tasks", task,
            "--timestamp", run_id,
        ]
        print(f"  [eval] RAGChecker for {task}/{run_id}")
        subprocess.run(cmd, cwd=str(DATA_PATH), check=False)

        after_eval = set(self._results_dir.glob("eval_results_*.json"))
        after_ea   = set(self._results_dir.glob("error_analysis_results_*.json"))

        new_eval = _newest(after_eval - before_eval)
        new_ea   = _newest(after_ea   - before_ea)

        # Attach scores to record
        records = _attach_scores(records, new_eval, new_ea)
        rec = records[0]

        # Parse agent cost from log file
        log_rel = rec.get("log_path", "")
        cost_info: dict = {}
        if log_rel:
            log_file = DATA_PATH / log_rel
            # log path: log/{agent}/{model}/{task}/{run_id}/log.log
            parts = Path(log_rel).parts
            log_agent = parts[1] if len(parts) > 1 else agent
            log_model = parts[2] if len(parts) > 2 else model
            cost_info = parse_run_cost(log_file, log_agent, log_model)
            print_cost(cost_info, log_agent, log_model)

        packet = {
            "task":             task,
            "log_path":         log_rel,
            "repo_path":        rec.get("repo_path"),
            "timestamp_id":     run_id,
            "score":            rec.get("score", {}),
            "agent_conclusion": rec.get("agent_conclusion", ""),
            "cost":             cost_info,
            "ran_at":           datetime.now().isoformat(),
        }
        return packet

    def primary_score(self, packet: dict) -> float:
        return float(packet.get("score", {}).get("f1") or 0.0)

    @property
    def log_root(self):
        return DATA_PATH

    def post_init_node(self, task: str, node) -> None:
        """No-op during experimentation — search tree is the source of truth."""

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_new_logs(
        self,
        tasks: list[str],
        agent: str,
        model: str,
        since: float,
        run_id: str | None = None,
    ) -> list[dict]:
        model_variants = {model, model.replace("/", "-"), model.split("/")[-1], "*"}
        seen:    set[str] = set()
        records: list[dict] = []

        for mv in model_variants:
            pattern = f"{agent}/{mv}"
            for log_file in self._log_dir.glob(f"{pattern}/*/*/log.log"):
                task = log_file.parts[-3]
                if tasks != ["*"] and task not in tasks:
                    continue
                ts_id = log_file.parent.name
                if run_id and ts_id != run_id:
                    continue
                if log_file.stat().st_mtime < since - 2:
                    continue
                try:
                    rel = str(log_file.relative_to(DATA_PATH))
                except ValueError:
                    rel = str(log_file)
                if rel in seen:
                    continue
                seen.add(rel)
                repo = _find_repo(log_file, self._runs_dir)
                try:
                    repo_rel = str(repo.relative_to(DATA_PATH)) if repo else None
                except ValueError:
                    repo_rel = str(repo) if repo else None
                records.append({
                    "task": task,
                    "log_path": rel,
                    "repo_path": repo_rel,
                    "timestamp_id": ts_id,
                    "score": {},
                })
        return records


# ---------------------------------------------------------------------------
# Module-level helpers (shared with attach_scores)
# ---------------------------------------------------------------------------

def _newest(paths: set[Path]) -> Path | None:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def _find_repo(log_path: Path, runs_dir: Path) -> Path | None:
    run_key = log_path.parent.name
    for d in runs_dir.iterdir():
        if d.is_dir() and run_key in d.name:
            return d
    return None


def _attach_scores(run_records: list[dict], eval_file: Path | None, ea_file: Path | None) -> list[dict]:
    """Attach RAGChecker scores to run_records.

    eval_file (eval_results_*.json) is required and provides precision/recall/f1.
    ea_file (error_analysis_results_*.json) is optional — when absent we
    position-match eval entries to records by index (eval.py is called with
    --timestamp so there's a 1-to-1 correspondence). FP/FN claim lists and
    agent_conclusion are intentionally NOT extracted here: those would leak
    ground-truth signal to the reflector.
    """
    if not eval_file:
        print("  [eval] No eval_results file — scores will be empty.")
        return run_records

    metric_re = re.compile(
        r'"precision"\s*:\s*([\d.]+).*?"recall"\s*:\s*([\d.]+).*?"f1"\s*:\s*([\d.]+)',
        re.DOTALL,
    )

    with open(eval_file, encoding="utf-8") as f:
        ev_data = json.load(f)

    ea_data: list = []
    if ea_file:
        try:
            with open(ea_file, encoding="utf-8") as f:
                ea_data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [eval] WARNING: ea_file unreadable ({e}); using eval_results alone")
            ea_data = []

    use_ea = bool(ea_data)

    matched = 0
    for i, rec in enumerate(run_records):
        ev_str = ""
        if use_ea:
            for j, ea in enumerate(ea_data):
                if ea.get("log_path") == rec["log_path"] and j < len(ev_data):
                    ev_str = ev_data[j]
                    break
        elif i < len(ev_data):
            ev_str = ev_data[i]

        ev_text = ev_str if isinstance(ev_str, str) else json.dumps(ev_str)
        m = metric_re.search(ev_text)
        # RAGChecker emits metrics on a 0-100 scale; normalise to 0-1 so they
        # match the autolab convention (and so prune_threshold/BAVT stats are
        # comparable across benchmarks).
        rec["score"] = {
            "precision": float(m.group(1)) / 100.0,
            "recall":    float(m.group(2)) / 100.0,
            "f1":        float(m.group(3)) / 100.0,
        } if m else {}
        rec["agent_conclusion"] = ""
        rec["false_positives"]  = []
        rec["false_negatives"]  = []
        rec["fp_results"]       = []
        rec["fn_results"]       = []
        if rec["score"]:
            matched += 1

    src = "ea-keyed" if use_ea else "position-matched"
    print(f"  [eval] Matched scores ({src}): {matched}/{len(run_records)}")
    return run_records
