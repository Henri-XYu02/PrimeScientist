"""
agents/codex/run_inherit.py — fire_bench codex runner with code inheritance.

Mirrors run.py but seeds the sandbox from a parent node's sandbox (set via
FIRE_BENCH_SEED_REPO) instead of copying pristine data + utils. The seed
already contains data/, utils/, and the agent's prior implementation, so the
child agent starts from where the parent left off and applies only the
incremental changes described in the staged skill.

Triggered by run_agent.py when FIRE_BENCH_SEED_REPO is set in env.
"""

import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

Main_Path = Path(__file__).parents[2]   # benchmarks/fire_bench/
# Unset-fallback kept identical to the harness (<repo>/fire_bench_data) so agent
# and harness always agree — see run.py for why a mismatch is catastrophic.
Data_Path = Path(os.environ.get("FIRE_BENCH_DATA") or (Main_Path.parent.parent / "fire_bench_data"))


def _load_skills(main_path: Path, task_id: str) -> str:
    """Build skill context (insights from other tasks + current task plan)."""
    parts = []

    insights_dir = main_path / "benchmark" / "insights"
    if insights_dir.exists():
        insight_parts = []
        for insight_file in sorted(insights_dir.glob("*.md")):
            if not insight_file.stat().st_size:
                continue
            if insight_file.stem == f"{task_id}_insight":
                continue
            content = insight_file.read_text(encoding="utf-8").strip()
            insight_parts.append(f"### {insight_file.stem}\n{content}")
        if insight_parts:
            parts.append(
                "## Insights from Other Tasks\n"
                "(Select what is relevant to your task — ignore the rest)\n\n"
                + "\n\n".join(insight_parts)
            )

    task_plan = main_path / "benchmark" / "papers" / task_id / "skill" / "skill.md"
    if task_plan.exists() and task_plan.stat().st_size > 0:
        parts.append("## Experimental Plan for This Task\n\n" + task_plan.read_text(encoding="utf-8").strip())

    return "\n\n---\n\n".join(parts)


def main() -> None:
    agent_id  = os.environ.get("AGENT_ID", "")
    task_id   = os.environ.get("TASK_ID", "")
    figure_id = os.environ.get("FIGURE_ID", "")
    LLM_MODEL = os.environ.get("LLM_MODEL", "")
    seed_repo = os.environ.get("FIRE_BENCH_SEED_REPO", "")

    seed_path = Path(seed_repo) if seed_repo else None
    if seed_path is None or not seed_path.exists():
        print(f"[run_inherit] ERROR: FIRE_BENCH_SEED_REPO missing or invalid: {seed_repo!r}")
        sys.exit(1)

    timestamp = time.strftime("%Y%m%d%H%M%S")
    rd = random.randint(10000, 99999)

    run_dir = Data_Path / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    sandbox_volume_filename = f"{agent_id}_{LLM_MODEL.replace('/','-')}_{timestamp}_{rd}"
    sandbox_volume_path = run_dir / sandbox_volume_filename


    # Seed from parent sandbox (contains data/, utils/, and the parent's code).
    print(f"[run_inherit] Seeding sandbox from {seed_path}")
    shutil.copytree(str(seed_path), str(sandbox_volume_path))

    # Copy this node's diff.py and changes.md into the sandbox so the agent
    # can read what differs from the parent's plan. node_dir is set by
    # fire_bench.run_agent; absent for non-CodeAct flows (root, etc.).
    node_dir_env = os.environ.get("FIRE_BENCH_NODE_DIR", "")
    if node_dir_env:
        node_dir = Path(node_dir_env)
        inherited_dir = sandbox_volume_path / "INHERITED_FROM_PARENT"
        inherited_dir.mkdir(exist_ok=True)
        copied = []
        for fname in ("diff.py", "changes.md"):
            src = node_dir / fname
            if src.exists():
                shutil.copy2(src, inherited_dir / fname)
                copied.append(fname)
        if copied:
            print(f"[run_inherit] Copied {copied} from {node_dir} to {inherited_dir}")

    # Instruction (same task → same instruction)
    INSTRUCTION_PATH = Main_Path / "benchmark" / "papers" / task_id / (figure_id if figure_id else "") / "instruction"
    instruction_file = INSTRUCTION_PATH / "instruction.txt"

    log_file = Data_Path / "log" / f"{agent_id}" / f"{LLM_MODEL}" / f"{task_id}" / f"{timestamp}_{rd}" / "log.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w") as f:
        f.write(f"agent_id: {agent_id}\n")
        f.write(f"task_id: {task_id}\n")
        f.write(f"llm_model: {LLM_MODEL}\n")
        f.write(f"seeded_from: {seed_path}\n")
        f.write("=" * 40 + "\n")

    with open(instruction_file, "r") as f:
        instruction_text = f.read().strip()

    # Prepend an inheritance notice so the agent's very first read makes it
    # clear the workspace is NOT pristine — the parent's scripts and outputs
    # are already present and should be updated, not re-created from scratch.
    has_changes_md = (sandbox_volume_path / "INHERITED_FROM_PARENT" / "changes.md").exists()
    if has_changes_md:
        inheritance_notice = (
            "NOTE — INHERITED WORKSPACE: this sandbox is seeded from a parent "
            "run's best implementation. Read `INHERITED_FROM_PARENT/changes.md` "
            "first to see exactly what differs from the parent's plan, then "
            "modify the existing code accordingly. Do NOT re-implement steps "
            "the parent already completed correctly — only apply the changes "
            "described in the updated experimental plan.\n\n"
        )
    else:
        inheritance_notice = (
            "NOTE — INHERITED WORKSPACE: this sandbox is seeded from a prior "
            "run's best implementation. Build on top of the existing code; do "
            "NOT re-implement steps already completed correctly — apply only "
            "the new improvements described in the updated experimental plan.\n\n"
        )
    instruction_text = inheritance_notice + instruction_text

    use_subscription = os.environ.get("USE_SUBSCRIPTION", "0") == "1"
    openai_key    = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    google_key    = os.environ.get("GOOGLE_API_KEY", "")
    hf_token      = os.environ.get("HF_TOKEN", "")

    # Overwrite the staged skill — parent's skill is in the seed; ours differs.
    skill_body = _load_skills(Main_Path, task_id)
    if skill_body:
        codex_skill_dir = sandbox_volume_path / ".agents" / "skills" / "research-agent-skill"
        codex_skill_dir.mkdir(parents=True, exist_ok=True)
        with open(codex_skill_dir / "SKILL.md", "w") as f:
            f.write("---\n")
            f.write("name: research-agent-skill\n")
            f.write('description: "Apply when conducting research experiments, replicating academic papers, running code experiments, or drawing conclusions from experimental data. Always apply for any research or scientific task."\n')
            f.write("---\n\n")
            f.write(skill_body)
        openai_yaml_dir = codex_skill_dir / "agents"
        openai_yaml_dir.mkdir(exist_ok=True)
        with open(openai_yaml_dir / "openai.yaml", "w") as f:
            f.write("interface:\n")
            f.write("  display_name: Research Agent Skill\n")
            f.write("  short_description: Guidelines and experimental plan for this task\n")
            f.write("policy:\n")
            f.write("  allow_implicit_invocation: true\n")
        print(f"[skill] Wrote Codex skill to {codex_skill_dir}")

    # Refresh API keys (in case the seed had stale ones)
    with open(sandbox_volume_path / ".env", "w") as f:
        f.write(f"OPENAI_API_KEY={openai_key}\n")
        f.write(f"ANTHROPIC_API_KEY={anthropic_key}\n")
        f.write(f"GOOGLE_API_KEY={google_key}\n")
        f.write(f"HF_TOKEN={hf_token}\n")

    # See agents/codex/run.py for the two execution modes (Docker vs. host).
    use_docker = os.environ.get("FIREBENCH_USE_DOCKER", "0") == "1"

    if use_docker:
        image = os.environ.get("FIREBENCH_CODEX_IMAGE", "firebench-codex:0.1")

        openai_env_keys = [
            "OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_HOST",
            "OPENAI_ORG", "OPENAI_ORG_ID", "OPENAI_PROJECT",
        ]
        openai_env_args: list[str] = []
        for k in openai_env_keys:
            v = os.environ.get(k)
            if v:
                openai_env_args.extend(["-e", f"{k}={v}"])

        cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{sandbox_volume_path}:/sandbox",
            "-v", "/home/xinle/.codex:/agent-home/.codex",
            "-w", "/sandbox",
            "-e", "HOME=/agent-home",

            # 🔥 core keys
            "-e", f"OPENAI_API_KEY={openai_key}",
            "-e", f"ANTHROPIC_API_KEY={anthropic_key}",
            "-e", f"GOOGLE_API_KEY={google_key}",
            "-e", f"HF_TOKEN={hf_token}",

            "-e", f"AGENT_ID={agent_id}",
            "-e", f"TASK_ID={task_id}",
            "-e", f"LLM_MODEL={LLM_MODEL}",
            image,
            "codex", "--model", LLM_MODEL, "--config", "sandbox_mode=danger-full-access", 
            "--ask-for-approval=never", "exec", "--skip-git-repo-check", instruction_text,
        ]
        run_kwargs = {}
        mode_label = f"Docker ({image})"
    else:
        cmd = [
            "npx", "@openai/codex@0.39.0",
            "--model", LLM_MODEL,
            "--config", "sandbox_mode=workspace-write",
            "--ask-for-approval=never",
            "exec", "--skip-git-repo-check",
            instruction_text,
        ]
        pip_scratch = Path(f"/tmp/firebench_pip_{timestamp}_{rd}")
        pip_scratch.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONUSERBASE"] = str(pip_scratch)
        env["PIP_USER"] = "1"
        run_kwargs = {"cwd": sandbox_volume_path, "env": env}
        mode_label = "host (sandbox_mode=workspace-write)"

    if use_subscription:
        print("Subscription is only available for Claude Code right now, still need OPENAI_API_KEY.")
    else:
        print(f"Running Codex with API key authentication [{mode_label}]")

    with open(log_file, "a") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, **run_kwargs)

    if not use_docker:
        try:
            shutil.rmtree(pip_scratch, ignore_errors=True)
        except NameError:
            pass

    print(f"Run complete. Logs saved to {log_file}")


if __name__ == "__main__":
    main()
