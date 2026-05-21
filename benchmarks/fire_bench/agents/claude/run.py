import subprocess
import os
import time
from pathlib import Path
import shutil
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Repo path
Main_Path = Path(__file__).parents[2]  # benchmarks/fire_bench/ (contains benchmark/ data)
Data_Path = Path("/data/xinle/FIRE-Bench")

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
    LLM_MODEL = os.environ.get("LLM_MODEL", "claude-3-5-sonnet-20240620")  # default fallback
    
    # Timestamp for run naming
    timestamp = time.strftime("%Y%m%d%H%M%S")
    run_name = f"claude-code-run-{timestamp}"

    # Ensure run/ directory exists
    run_dir = Data_Path / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build the sandbox directory
    sandbox_volume_filename = f"{agent_id}_{LLM_MODEL.replace('/','-')}_{timestamp}"
    sandbox_volume_path = run_dir / sandbox_volume_filename

    # Copy experiment setup
    instances_src = (
        Main_Path / f"benchmark/papers/{task_id}/{figure_id}/data"
        if figure_id
        else Main_Path / f"benchmark/papers/{task_id}/data"
    )
    shutil.copytree(instances_src, sandbox_volume_path)
    utils_src = Main_Path / "utils"
    shutil.copytree(utils_src, sandbox_volume_path / "utils")

    # Instruction path
    INSTRUCTION_PATH = Main_Path / "benchmark" / "papers" / task_id / (figure_id if figure_id else "") / "instruction"
    instruction_file = INSTRUCTION_PATH / "instruction.txt"

    # Log file
    log_file = Data_Path / "log" / f"{agent_id}" / f"{LLM_MODEL}" / f"{task_id}" / f"{timestamp}" / "log.log"
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

    # Write skills as CLAUDE.md into sandbox — Claude Code auto-loads it from cwd.
    skill_body = _load_skills(Main_Path, task_id)
    if skill_body:
        with open(sandbox_volume_path / "CLAUDE.md", "w") as f:
            f.write("# Research Agent Skill Guide\n\n")
            f.write("The following guidelines and plan were generated from past runs. ")
            f.write("Apply them throughout this task.\n\n")
            f.write(skill_body)
        print(f"[skill] Wrote CLAUDE.md to sandbox")

    # Write .env into sandbox so utils/llm_inference.py can load API keys
    # Note: Sandbox always gets API keys for running experiments, regardless of subscription mode
    with open(sandbox_volume_path / ".env", "w") as f:
        f.write(f"OPENAI_API_KEY={openai_key}\n")
        f.write(f"ANTHROPIC_API_KEY={anthropic_key}\n")
        f.write(f"GOOGLE_API_KEY={google_key}\n")
        f.write(f"HF_TOKEN={hf_token}\n")

    # Build Claude Code CLI command
    cmd = [
        "claude",
        "-p", instruction_text,
        "--model", LLM_MODEL,
        "--output-format", "stream-json",   # trajectory output
        "--verbose",                        # required for stream-json
        "--add-dir", str(sandbox_volume_path),
        "--add-dir", str(sandbox_volume_path / "utils"),
        "--dangerously-skip-permissions"
    ]

    # Prepare environment for Claude CLI
    env = os.environ.copy()
    if use_subscription:
        # Remove ANTHROPIC_API_KEY from Claude CLI's environment so it uses subscription
        # Sandbox experiments still have access via the .env file we wrote above
        env.pop("ANTHROPIC_API_KEY", None)
        print(f"Running Claude Code in subscription mode (experiments use API key from sandbox .env)")
    else:
        print(f"Running Claude Code with API key")


    # Run the Claude Code command and log output
    with open(log_file, "a") as f:
        process = subprocess.run(cmd, cwd=sandbox_volume_path, env=env, stdout=f, stderr=subprocess.STDOUT)

    print(f"Run complete. Logs saved to {log_file}")

if __name__ == "__main__":
    main()
