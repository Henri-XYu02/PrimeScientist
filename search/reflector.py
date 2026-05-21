"""
search/reflector.py — Coding-agent reflector for FIRE-Bench skill search.

The reflector is a full Codex / Claude Code agent session — not a raw LLM API
call. It receives the search tree directory and a minimal prompt, then reads
whatever files it finds relevant (logs, prior skill.md, packets, prior.json)
via standard shell tools (cat, grep, ls). This is the Meta-Harness approach:
full filesystem access instead of pre-digested summaries.

Session persistence:
  - Claude Code: `--resume <sessionId>` reuses the same conversation across calls,
    giving the agent memory of every prior proposal it made.
  - Codex: fresh session each call, but reads prior proposals from the tree itself
    (skill.md, prior.json, stats.json files serve as its external memory).

Session ID is stored at: search_tree/{task}/reflector_session.json
"""

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from benchmarks.utils import parse_run_cost, print_cost, append_cost_ledger

SESSION_FILE      = "reflector_session.json"
# Dot-prefixed so it is hidden from default `ls` — reflector subprocesses
# browsing the tree don't see stale reflector_log.log files and spend tokens
# re-reading them. Visible still with `ls -a`.
REFLECTOR_LOG_DIR = ".reflector_logs"



def _write_reflector_log(cwd: str, label: str, stdout: str, stderr: str) -> Path:
    """Write reflector stdout+stderr to a timestamped log file and return the path."""
    log_dir = Path(cwd) / REFLECTOR_LOG_DIR
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{label}_{ts}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        if stdout:
            f.write("=== STDOUT ===\n")
            f.write(stdout)
            f.write("\n")
        if stderr:
            f.write("=== STDERR ===\n")
            f.write(stderr)
            f.write("\n")
    print(f"  [reflector] log → {log_path}")
    return log_path

# ---------------------------------------------------------------------------
# Prompts  (filesystem-first: tell the agent WHERE to look, not WHAT to see)
# ---------------------------------------------------------------------------

INIT_PROMPT = """\
You are initialising a new task: {task_id}

Your working directory contains:
  instruction{instruction_suffix}  — task description (read this first)
  insights/                        — real discoveries from OTHER completed tasks
                                     (may be empty if this is the first task)

Your job — write ONE file into root/:
  root/skill.md   (< 10 KB)

      The content depends on the task type:

      For RESEARCH REPLICATION tasks (the instruction asks you to replicate an experiment):
        - exact datasets (name, source, sample size as stated in the instruction)
        - exact method (model, hyperparameters, evaluation metric)
        - decide any unspecified hyperparameters
        - make the plan such that each run can be completed in roughly 1 hour
        - step-by-step reproduction procedure

      For CODE OPTIMISATION tasks (the instruction asks you to optimise code for speed/efficiency):
        - which files to edit and which to leave read-only
        - concrete optimisation strategies to try (data structures, algorithms, concurrency)
        - implementation order (highest-impact changes first)
        - correctness constraints to keep in mind

Do NOT write root/{task_id}_insight.md — it starts empty and is filled from real results.
Do NOT invent outcomes — only plan based on what the instruction and insight files say.
Do NOT leave ambiguities — decide anything that is underspecified.

Steps:
1. cat instruction{instruction_suffix}
2. ls insights/ — read any files that look relevant to this task
3. Write root/skill.md
"""

# Injected into {pruned_section} only when all previous children were pruned.
_PRUNED_SECTION = """\
  {{node_rel}}/pruned_children.md — READ THIS FIRST. All previous children of this
                                 node were pruned (score fell too far below the
                                 parent). Each entry has the prior.json hypothesis,
                                 rationale, and verifier verdict.
                                 You MUST NOT repeat any of those approaches.
                                 Check `ls {{node_rel}}/children/` so your new
                                 proposals use names that don't conflict with
                                 existing dirs (including hidden .pruned_* ones).
"""

REFLECT_PROMPT = """\
You are a reflector for task: {task_id}

A new run has just completed. Your working directory is the search tree root for this task.
The completed node is at: {node_rel}/

If this is not your first call on this task, you are RESUMING a previous
session — you already have memory of files and proposals you inspected last
time. Do NOT re-read files whose content you already remember; just check for
NEW files (e.g. the newly-completed node's packet.json, its diff, its verifier
log) and whatever sibling state has changed since your last call.

Relevant files you should read (anything not listed here is noise — skip it):
{pruned_section}  {node_rel}/packet.json       — score and run metadata.
                                 For runs that failed (reward=0), inspect
                                 `reward_detail.verifier_output_tail` inside this
                                 packet — it contains the verifier's build errors
                                 and/or correctness-diff output, which tells you
                                 EXACTLY why the run failed. Without this you'll
                                 just be guessing.
  {node_rel}/sandbox/_reward/diff_from_baseline.log
                               — compact `diff -ruN environment/ sandbox/` showing
                                 EXACTLY which lines the agent changed relative to
                                 the pristine baseline. READ THIS FIRST if you need
                                 to know what the agent did. Do NOT `cat` every
                                 source file in sandbox/.
  {node_rel}/sandbox/_reward/verifier.log
                               — full verifier stdout/stderr if the tail in
                                 packet.json is truncated. Read only if the tail
                                 doesn't give enough context.
  {node_rel}/log_brief.log     — head+tail excerpt of the agent run (the full log is
                                 intentionally hidden to keep this reflection cheap —
                                 the brief is what you work with). Note: the agent
                                 often *claims* correctness in its summary even when
                                 the verifier disagrees — TRUST the verifier output,
                                 not the agent's narrative.
  {node_rel}/skill.md          — the skill this run executed
  {node_rel}/sandbox/          — the agent's run workspace. Do NOT recurse this
                                 tree; use diff_from_baseline.log to find what
                                 changed, then read ONLY the specific files the
                                 diff points to if more context is needed.
  Other nodes' prior.json / skill.md / packet.json in the tree — when you need to
                                 know previous trials' approach and performance

Your job:
1. Read packet.json. If reward is 0, START with
   `reward_detail.verifier_output_tail` — it usually names the exact file/line
   where the build broke or the first divergent line of the correctness diff.
   Then read {node_rel}/log_brief.log to correlate with what the agent tried.
   Optionally inspect specific files in {node_rel}/sandbox/ that the agent changed.
   Ground your analysis in what the verifier actually reported, not just the score
   or the agent's own claims.
2. Check other nodes' prior.json / skill.md / packet.json as needed to avoid re-proposing a hypothesis
   that's already been tried, or to learn previous trials' performance to better reflect.
3. Extract experimental setup failures, or factors in skill file that cause runtime failures.

4. Propose between 1 and {n_proposals} CONTROVERSIALLY DIFFERENT skill variants.
   "Controversially different" means each proposal bets on a fundamentally different
   hypothesis — e.g. different algorithm, different data structure, different trade-off,
   different part of the code to change. Do NOT create proposals that differ only in
   a parameter value or a minor wording change; those are not separate bets.

   You decide how many to create:
   - If you see one clearly dominant direction: write 1 proposal.
   - If you see 2–{n_proposals} genuinely competing hypotheses each worth testing: write that many.
   - Never pad with minor variations just to hit a count.

{proposal_block}
Print a one-line summary for each proposal and each pruned branch.
"""

# Variant of REFLECT_PROMPT for research-replication benchmarks (e.g. fire_bench)
# where there is no _reward directory, diff_from_baseline.log, or verifier.log.
# Instead the agent leaves experiment result files in sandbox/ subdirectories.
FIRE_BENCH_REFLECT_PROMPT = """\
You are a reflector for task: {task_id}

A new run has just completed. Your working directory is the search tree root for this task.
The completed node is at: {node_rel}/

If this is not your first call on this task, you are RESUMING a previous
session — you already have memory of files and proposals you inspected last
time. Do NOT re-read files whose content you already remember; just check for
NEW files (e.g. the newly-completed node's packet.json, result directories) and
whatever sibling state has changed since your last call.

Relevant files you should read (anything not listed here is noise — skip it):
{pruned_section}  {node_rel}/packet.json       — reward score (precision/recall/f1) and run metadata, the reward score symbolizes the correctness of agent run conclusion.
  {node_rel}/log_brief.log     — head+tail excerpt of the agent run. This is your
                                 main diagnostic tool: look for Python tracebacks,
                                 missing-file errors, or the agent admitting it
                                 could not finish. The agent often *claims* success
                                 — trust packet.json score, not the agent's narrative.
  {node_rel}/.log.log          — FULL agent log (hidden, very large). Read this ONLY
                                 if log_brief.log leaves the failure cause genuinely
                                 ambiguous — e.g. the tail ends mid-traceback and you
                                 cannot identify the root cause. Do NOT read it
                                 routinely; prefer the brief; Read it ONLY WHEN the reward score is very low.
  {node_rel}/skill.md          — the experimental plan this run executed
  {node_rel}/changes.md        — (if present) short prose summary of what this
                                 node's plan changed relative to its parent.
                                 Read this to quickly understand the intended delta
                                 without diffing the full skill.md files.
  {node_rel}/sandbox/          — the agent's experiment workspace. Use `ls` to
                                 discover result directories (e.g. results_*/),
                                 then read specific output files only if you need
                                 to understand what the agent actually produced.
                                 Do NOT cat every source file; use targeted reads.
  Other nodes' prior.json / skill.md / packet.json in the tree — when you need to
                                 know previous trials' approach and performance

Your job:
1. Read packet.json for the score. You can read log_brief.log to find the agent run trajectory.
   Use `ls {node_rel}/sandbox/` to see what result directories exist, then
   read specific files if the log is not enough to diagnose the failure.
   Ground your analysis in concrete observations, not assumptions.
2. Check other nodes' prior.json / skill.md / packet.json as needed to avoid
   re-proposing hypotheses already tried, and to learn what has and hasn't worked.
3. Identify whether failure was due to a setup/runtime error (fixable by changing
   the experimental plan) or a genuine result (the hypothesis was tested but scored low).

4. Propose between 1 and {n_proposals} CONTROVERSIALLY DIFFERENT skill variants.
   "Controversially different" means each proposal bets on a fundamentally different
   hypothesis — e.g. different hyperparameter choices, different metric
   interpretation, different dataset size and choices, etc. Do NOT create
   proposals that differ only in a minor hyperparameter value or a minor wording change;
   those are not separate bets.

   You decide how many to create:
   - If you see one clearly dominant direction: write 1 proposal.
   - If you see 2–{n_proposals} genuinely competing hypotheses each worth testing: write that many.
   - Never pad with minor variations just to hit a count.

{proposal_block}
Print a one-line summary for each proposal and each pruned branch.
"""

# Default proposal-creation block — direct skill.md write. Suitable for
# autolab where the skill is a high-level pointer and code is the source of truth.
DEFAULT_PROPOSAL_BLOCK = """\
   Create child directories numbered from where you left off:
     {node_rel}/children/proposal_0/
     {node_rel}/children/proposal_1/
     ... (up to proposal_{n_proposals_minus_1}/)
   In each child directory write TWO files:
   a) skill.md (< 5 KB) — the plan the child agent will execute. Read the
      parent's skill.md ({node_rel}/skill.md) first for context, then write
      ONLY what the child agent needs to act on. If code inheritance is
      active (see note below), the child's sandbox already contains the
      parent's code — write a compact "Inherited state" summary (3-5 bullets
      of what the parent already implemented) followed by a "This proposal"
      section describing only the additional incremental change. Do not
      copy unchanged sections verbatim from the parent skill.
   b) prior.json with structured, detailed reasoning. This is the durable
      record the NEXT reflector will consult instead of your (now-discarded)
      thinking — so invest your analysis here rather than in stdout:
        {{
          "estimate": <float 0-1>,
          "hypothesis": "<the single concrete hypothesis this proposal tests>",
          "rationale": "<2-3 sentences: WHY you expect this to improve the score,
                         grounded in specific observations from packet.json /
                         log_brief.log / sandbox — cite file or section>",
          "changes": "<1 sentences naming which files/sections this proposal
                       modifies, at what abstraction level (algorithm / data
                       structure / hyperparam)>",
          "risks": "<1-2 sentences on the most plausible failure mode>"
        }}
"""

# CodeAct-style block — diff.py transforms parent skill.md to child's full
# plan, plus a short changes.md the agent reads in the inherited sandbox.
# Suitable for fire_bench where each child needs a complete experimental plan.
CODEACT_PROPOSAL_BLOCK = """\
   Create child directories numbered from where you left off:
     {node_rel}/children/proposal_0/
     {node_rel}/children/proposal_1/
     ... (up to proposal_{n_proposals_minus_1}/)
   In each child directory write THREE files (and run the diff):
   a) diff.py — a self-contained Python script that reads the parent's
      skill.md and writes a NEW COMPLETE skill.md into this directory.
      The output skill.md must be a FULL self-contained experimental plan
      (dataset, model, hyperparameters, evaluation metrics, conclusion
      structure) — NOT a delta. Template:
        from pathlib import Path
        parent = Path(__file__).parents[2] / "skill.md"
        output = Path(__file__).parent / "skill.md"
        text = parent.read_text()
        # Apply targeted transformations to produce the new full plan,
        # e.g. text = text.replace("model = X", "model = Y")
        output.write_text(text)
   b) Run it immediately: python diff.py
      Verify skill.md was created and is the full updated plan.
      If diff.py errors, fix it and re-run.
   c) changes.md (< 1 KB) — short prose summary of what changed compared to
      the parent's plan. 3-6 bullet points written FOR the child agent
      (e.g. "switched dataset from X to Y", "moved evaluation from accuracy
      to F1", "added an ablation on hyperparameter Z"). The child agent
      reads this file in its inherited sandbox to know what to update
      without re-doing the parent's work.
   d) prior.json with structured, detailed reasoning. This is the durable
      record the NEXT reflector will consult:
        {{
          "estimate": <float 0-1>,
          "hypothesis": "<the single concrete hypothesis this proposal tests>",
          "rationale": "<2-3 sentences: WHY you expect this to improve the score,
                         grounded in specific observations from packet.json /
                         log_brief.log / sandbox — cite file or section>",
          "changes": "<1 sentences naming which files/sections this proposal
                       modifies, at what abstraction level (algorithm / data
                       structure / hyperparam)>",
          "risks": "<1-2 sentences on the most plausible failure mode>"
        }}

   Do NOT write skill.md directly — only via diff.py.
"""

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _load_session_state(session_file: Path) -> dict:
    """Return the full session state dict (session_id + optional extras like
    codex_last_cumulative_tokens). Returns {} if the file doesn't exist or is
    unreadable."""
    if session_file.exists():
        try:
            return json.loads(session_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _load_session(session_file: Path) -> str | None:
    """Shortcut: just the session_id, for callers that don't care about extras."""
    return _load_session_state(session_file).get("session_id")


def _save_session_state(session_file: Path, **updates) -> None:
    """Merge-write: preserve existing keys, overwrite any provided in updates."""
    state = _load_session_state(session_file)
    state.update(updates)
    session_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _save_session(session_file: Path, session_id: str) -> None:
    """Backward-compat wrapper: save only the session_id field."""
    _save_session_state(session_file, session_id=session_id)


def _parse_claude_model(output: str) -> str | None:
    """Extract model name from Claude Code stream-json output (modelUsage keys)."""
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # result objects have modelUsage: {"claude-opus-4-6": {...}}
            model_usage = obj.get("modelUsage") or {}
            if model_usage:
                return next(iter(model_usage))
        except json.JSONDecodeError:
            pass
    return None


def _parse_codex_model(output: str) -> str | None:
    """Extract model name from codex header line 'model: <name>'."""
    m = re.search(r"^model:\s*(\S+)", output, re.MULTILINE)
    return m.group(1) if m else None


def _parse_codex_session_id(output: str) -> str | None:
    """Extract session id from codex banner: 'session id: <UUID>'."""
    m = re.search(r"^session id:\s*([0-9a-fA-F-]{36})", output, re.MULTILINE)
    return m.group(1) if m else None


def _parse_session_id(output: str) -> str | None:
    """Extract sessionId from Claude Code stream-json output."""
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # Claude Code emits {"type":"system","subtype":"init","session_id":"..."}
            if obj.get("type") == "system" and "session_id" in obj:
                return obj["session_id"]
            # Also appears in result objects
            if "session_id" in obj:
                return obj["session_id"]
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

def _run_claude_code(
    prompt: str,
    cwd: str,
    session_file: Path,
    timeout: int = 600,
) -> tuple[bool, dict]:
    """Run Claude Code as the reflector. Returns (success, cost_info)."""
    session_id = _load_session(session_file)

    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--verbose",
    ]
    if session_id:
        cmd += ["--resume", session_id, "--print", prompt]
    else:
        cmd += ["--print", prompt]

    print(f"  [reflector:claude] cwd={cwd}  session={session_id or 'new'}")
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  [reflector:claude] TIMEOUT after {timeout}s")
        return False, {}

    log_path = _write_reflector_log(cwd, "claude", result.stdout or "", result.stderr or "")
    reflector_model = _parse_claude_model(result.stdout or "") or "claude-opus-4-6"
    cost_info = parse_run_cost(log_path, "claude", reflector_model)
    print_cost(cost_info, "reflector:claude", reflector_model)
    append_cost_ledger(Path(cwd) / "costs.tsv", cost_info, "reflector:claude", reflector_model)

    if result.stdout:
        sid = _parse_session_id(result.stdout)
        if sid:
            _save_session(session_file, sid)
            print(f"  [reflector:claude] session_id saved: {sid}")

    if result.returncode != 0:
        print(f"  [reflector:claude] exit {result.returncode}")
        if result.stderr:
            print(result.stderr[:500])
        return False, cost_info

    return True, cost_info


REFLECTOR_CODEX_VERSION = "0.121.0"


def _run_codex(
    prompt: str,
    cwd: str,
    session_file: Path,
    model: str = "o4-mini",
    timeout: int = 600,
) -> tuple[bool, dict]:
    """
    Run Codex as the reflector via npx (pinned version, avoids global 0.39.0 TTY issues).
    Returns (success, cost_info).

    Session persistence: after each call we capture the `session id: UUID`
    from codex's banner and save it to SESSION_FILE. Subsequent calls in the
    same task dir use `exec resume --skip-git-repo-check <UUID> <prompt>` so
    the reflector keeps its memory of previously-inspected files and proposals.

    IMPORTANT — cumulative-tokens handling: codex's banner reports
    `tokens used: N` as a CUMULATIVE count for the whole session, not for
    just this resume. Naively recording N each call would double/triple-count
    and collapse the budget estimator. We fix this by persisting the previous
    call's cumulative in SESSION_FILE and charging only the delta (new_cumul
    minus last_cumul) to this call's cost ledger entry.
    """
    state          = _load_session_state(session_file)
    session_id     = state.get("session_id")
    last_cumul     = int(state.get("codex_last_cumulative_tokens", 0))

    base_cmd = [
        "npx", "--yes", f"@openai/codex@{REFLECTOR_CODEX_VERSION}",
        "--model", model,
        "--config", "sandbox_mode=danger-full-access",
        "--ask-for-approval=never",
    ]

    if session_id:
        # NB: --skip-git-repo-check must be included here too — without it,
        # codex refuses to run in non-git directories even when resuming.
        cmd = base_cmd + ["exec", "resume", "--skip-git-repo-check",
                          session_id, prompt]
        print(f"  [reflector:codex] resuming session={session_id[:8]}…  "
              f"last_cumul={last_cumul:,}  (codex@{REFLECTOR_CODEX_VERSION} model={model})")
    else:
        cmd = base_cmd + ["exec", "--skip-git-repo-check", prompt]
        print(f"  [reflector:codex] starting fresh session "
              f"(codex@{REFLECTOR_CODEX_VERSION} model={model})")

    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  [reflector:codex] TIMEOUT after {timeout}s")
        return False, {}

    # If we asked to resume and codex failed (e.g. saved session is stale or
    # was garbage-collected), drop the stored state so the next attempt
    # starts fresh instead of looping on the bad id.
    if session_id and result.returncode != 0:
        print(f"  [reflector:codex] resume failed (exit {result.returncode}) — "
              f"clearing saved session state")
        session_file.unlink(missing_ok=True)

    log_path = _write_reflector_log(cwd, "codex", result.stdout or "", result.stderr or "")
    reflector_model = _parse_codex_model(result.stdout or "") or model
    cost_info = parse_run_cost(log_path, "codex", reflector_model)

    # Subtract previous cumulative so this row records only THIS call's spend.
    if cost_info:
        new_cumul = int(cost_info.get("input_tokens", 0))
        delta     = max(0, new_cumul - last_cumul)
        if session_id and delta != new_cumul:
            cost_info = dict(cost_info)
            cost_info["input_tokens"] = delta
            in_p, _out_p = _openai_unit_price(reflector_model)
            cost_info["cost_usd"] = round(delta * in_p / 1_000_000, 6)
            print(f"  [reflector:codex] cumulative {new_cumul:,} − prev {last_cumul:,} "
                  f"= delta {delta:,} tokens (charged to this call)")
    else:
        new_cumul = last_cumul

    print_cost(cost_info, "reflector:codex", reflector_model)
    append_cost_ledger(Path(cwd) / "costs.tsv", cost_info, "reflector:codex", reflector_model)

    # Capture session id + updated cumulative for the next call.
    if result.stdout:
        sid = _parse_codex_session_id(result.stdout)
        if sid:
            _save_session_state(
                session_file,
                session_id=sid,
                codex_last_cumulative_tokens=new_cumul,
            )
            if sid != session_id:
                print(f"  [reflector:codex] session_id saved: {sid[:8]}…")

    if result.returncode != 0:
        print(f"  [reflector:codex] exit {result.returncode}")
        if result.stderr:
            print(result.stderr[:500])
        return False, cost_info

    return True, cost_info


def _run_reflector_agent(
    prompt: str,
    cwd: str,
    session_file: Path,
    agent: str = "claude",
    model: str = "o4-mini",
    timeout: int = 600,
) -> tuple[bool, dict]:
    """Returns (success, cost_info)."""
    if agent == "claude":
        return _run_claude_code(prompt, cwd, session_file, timeout)
    elif agent == "codex":
        return _run_codex(prompt, cwd, session_file, model, timeout)
    else:
        raise ValueError(f"Unknown reflector agent: {agent!r} (choose 'claude' or 'codex')")


# ---------------------------------------------------------------------------
# Dataset samples hint (minimal — agent reads tree itself, but we tell it
# where to look for data files)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_task(
    task: str,
    root_node,            # TreeNode
    instruction_path: Path,
    insights_dir: Path | None = None,
    agent: str = "claude",
    model: str = "o4-mini",
    tree_dir: Path | None = None,
    timeout: int = 600,
) -> bool:
    """
    Initialise root/skill.md for a brand-new task.
    The insight file starts empty — it is populated from real run results only.
    Cross-task insights from other completed tasks are copied in as reference.
    """
    cwd_path = tree_dir or root_node.path.parent
    cwd_path.mkdir(parents=True, exist_ok=True)

    # Copy instruction into tree_dir
    instruction_suffix = instruction_path.suffix or ".txt"
    shutil.copy2(instruction_path, cwd_path / f"instruction{instruction_suffix}")

    # Copy other tasks' real insight files (exclude this task's own — it hasn't run yet)
    local_insights = cwd_path / "insights"
    local_insights.mkdir(exist_ok=True)
    if insights_dir and insights_dir.exists():
        for f in insights_dir.glob("*.md"):
            if f.stem != f"{task}_insight":   # skip this task's own file
                shutil.copy2(f, local_insights / f.name)

    # Create empty insight file for this task — filled by reflector after real runs
    root_node.insight_path.touch()

    prompt = INIT_PROMPT.format(
        task_id=task,
        instruction_suffix=instruction_suffix,
    )

    session_file = cwd_path / SESSION_FILE
    ok, _ = _run_reflector_agent(prompt, str(cwd_path), session_file, agent=agent, model=model, timeout=timeout)
    return ok


LOG_BRIEF_HEAD_LINES = 100       # lines kept from the start of the agent log
LOG_BRIEF_TAIL_LINES = 500      # lines kept from the end


def _write_log_brief(full_log: Path, brief: Path,
                     head: int = LOG_BRIEF_HEAD_LINES,
                     tail: int = LOG_BRIEF_TAIL_LINES) -> None:
    """Write a head+tail excerpt of an agent log for the reflector.

    Saves the caller from loading 50-150k tokens of verbose thinking/tool-calls
    while preserving the task header (start) and the agent's conclusion (end).
    """
    try:
        lines = full_log.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return
    if len(lines) <= head + tail:
        brief.write_text("\n".join(lines), encoding="utf-8")
        return
    head_block = lines[:head]
    tail_block = lines[-tail:]
    omitted = len(lines) - head - tail
    sep = [
        "",
        f"... [{omitted} middle lines omitted — read log.log directly if needed] ...",
        "",
    ]
    brief.write_text("\n".join(head_block + sep + tail_block), encoding="utf-8")


def reflect_and_propose(
    task: str,
    root_node,           # TreeNode
    expand_node,         # TreeNode
    n_proposals: int,
    tree_dir: Path,
    insights_dir: Path,
    agent: str = "claude",
    model: str = "o4-mini",
    timeout: int = 600,
    is_reexpand: bool = False,
    inheritance_note: str = "",
    proposal_block: str = "",
    reflect_template: str = "",
) -> tuple[bool, dict]:
    """
    Reflect on the new run and propose N skill variants as child directories.
    Returns (success, cost_info) where cost_info is the reflector's token cost.
    """
    from search.tree import best_node

    # Keep local insights/ in sync with the benchmark insights_dir
    local_insights = tree_dir / "insights"
    local_insights.mkdir(exist_ok=True)
    if insights_dir.exists():
        for f in insights_dir.glob("*.md"):
            shutil.copy2(f, local_insights / f.name)

    best_q, _ = best_node(root_node)

    node_rel = expand_node.path.relative_to(tree_dir)

    # Produce a head+tail brief of the expanded node's agent log. The full log
    # is written as a dot-prefixed file (hidden from default `ls`) by
    # run_search._move_artifacts, so the reflector only sees the brief here.
    hidden_full = expand_node.path / ".log.log"
    legacy_full = expand_node.path / "log.log"     # pre-hiding name
    full_log = hidden_full if hidden_full.exists() else legacy_full
    if full_log.exists():
        _write_log_brief(full_log, expand_node.path / "log_brief.log")

    pruned_section = (
        _PRUNED_SECTION.format(node_rel=str(node_rel)) if is_reexpand else ""
    )
    block_template = proposal_block or DEFAULT_PROPOSAL_BLOCK
    base_template = reflect_template or REFLECT_PROMPT
    full_template = base_template.replace("{proposal_block}", block_template)
    prompt = full_template.format(
        task_id=task,
        node_rel=str(node_rel),
        n_proposals=n_proposals,
        n_proposals_minus_1=max(0, n_proposals - 1),
        pruned_section=pruned_section,
    )
    if inheritance_note:
        prompt += inheritance_note

    session_file = tree_dir / SESSION_FILE
    return _run_reflector_agent(prompt, str(tree_dir), session_file, agent=agent, model=model, timeout=timeout)
