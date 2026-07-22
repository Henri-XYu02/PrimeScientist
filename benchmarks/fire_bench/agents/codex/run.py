import subprocess
import os
import time
import random
from pathlib import Path
import shutil
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Repo path
Main_Path = Path(__file__).parents[2]  # benchmarks/fire_bench/ (contains benchmark/ data)
# Must match the harness (FireBench reads FIRE_BENCH_DATA). Hardcoding this broke
# per-repeat isolation: the agent wrote logs somewhere the harness never looked,
# so every run was reported "agent run failed" while still spending real tokens.
Data_Path = Path(os.environ.get("FIRE_BENCH_DATA") or "/data/xinle/FIRE-Bench")


def _load_skills(main_path: Path, task_id: str) -> str:
    """
    Build skill context for the agent:
      1. Relevant cross-task insight files from benchmark/insights/
         (all loaded; agent selects what applies — kept short by design)
      2. This task's experimental plan from benchmark/papers/{task}/skill/skill.md
    """
    parts = []

    insights_dir = main_path / "benchmark" / "insights"
    if insights_dir.exists():
        insight_parts = []
        for insight_file in sorted(insights_dir.glob("*.md")):
            if not insight_file.stat().st_size:
                continue
            # Skip the current task's own insight (included below via skill.md)
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


def main():
    # Prepare environment variables
    agent_id = os.environ.get("AGENT_ID", "")
    task_id = os.environ.get("TASK_ID", "")
    figure_id = os.environ.get("FIGURE_ID", "")
    LLM_MODEL = os.environ.get("LLM_MODEL", "")

    # Timestamp for run naming
    timestamp = time.strftime("%Y%m%d%H%M%S")
    run_name = f"codex-run-{timestamp}"

    # A random suffix
    rd = random.randint(10000, 99999)

    # Ensure run/ directory exists
    run_dir = Data_Path / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build the sandbox directory
    sandbox_volume_filename = f"{agent_id}_{LLM_MODEL.replace('/','-')}_{timestamp}_{rd}"
    sandbox_volume_path = run_dir / sandbox_volume_filename

    # Copy experiment setup
    instances_src = Main_Path / f"benchmark/papers/{task_id}/{figure_id}/data" if figure_id else Main_Path / f"benchmark/papers/{task_id}/data"
    shutil.copytree(instances_src, sandbox_volume_path)
    utils_src = Main_Path / "utils"
    shutil.copytree(utils_src, sandbox_volume_path / "utils")

    # Instruction path
    INSTRUCTION_PATH = Main_Path / "benchmark" / "papers" / task_id / (figure_id if figure_id else "") / "instruction"
    instruction_file = INSTRUCTION_PATH / "instruction.txt"

    # Log file
    log_file = Data_Path / "log" / f"{agent_id}" / f"{LLM_MODEL}" / f"{task_id}" / f"{timestamp}_{rd}" / "log.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Write metadata to log
    with open(log_file, "w") as f:
        f.write(f"agent_id: {agent_id}\n")
        f.write(f"task_id: {task_id}\n")
        f.write(f"llm_model: {LLM_MODEL}\n")
        f.write("=" * 40 + "\n")

    with open(instruction_file, "r") as f:
        instruction_text = f.read().strip()

    # Load API keys and settings from .env
    use_subscription = os.environ.get("USE_SUBSCRIPTION", "0") == "1"
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    # Inject skills as a Codex native skill directory in the sandbox.
    # Codex auto-discovers .agents/skills/ in the cwd and matches via description.
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

    # Write .env into sandbox so utils/llm_inference.py can load API keys
    with open(sandbox_volume_path / ".env", "w") as f:
        f.write(f"OPENAI_API_KEY={openai_key}\n")
        f.write(f"ANTHROPIC_API_KEY={anthropic_key}\n")
        f.write(f"GOOGLE_API_KEY={google_key}\n")
        f.write(f"HF_TOKEN={hf_token}\n")

    # Two execution modes:
    #   FIREBENCH_USE_DOCKER=1 → run codex inside a Docker container. The sandbox
    #     is mounted at /sandbox; everything else (pip installs, /tmp scratch)
    #     lives in the container's ephemeral writable layer and vanishes on
    #     --rm. Strongest isolation, but needs Docker root on /data.
    #   default → run codex on the host with sandbox_mode=workspace-write so it
    #     can only write inside the workspace, plus PIP_USER pointing at a
    #     scratch dir outside the sandbox (so agent pip installs don't bloat
    #     the sandbox or corrupt the host conda env).
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
            "--config", "sandbox_mode=workspace-write",     # block writes outside the sandbox
            "--ask-for-approval=never",
            "exec", "--skip-git-repo-check",
            instruction_text,
        ]
        # Redirect any pip --user installs to a per-run scratch dir outside
        # the sandbox; cleaned up at the end of the run.
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
