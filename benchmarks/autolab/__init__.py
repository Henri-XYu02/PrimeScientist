"""
benchmarks/autolab/__init__.py — AutoLab benchmark implementation.

Each task provides:
  - tasks/{task}/environment/   — editable codebase (copied to sandbox for agent)
  - tasks/{task}/instruction.md — agent-facing problem description
  - tasks/{task}/tests/test.sh  — verifier: builds code, runs correctness + benchmark,
                                  writes /logs/verifier/reward.json (reward 0–1)
  - tasks/{task}/task.toml      — resource limits (cpus, memory_mb)
  - tasks/{task}/solution/      — reference (NOT shown to agent or reflector)

Agent workflow:
  1. setup_node  — stage skill.md for run_agent to pick up
  2. run_agent   — copy environment/ to sandbox, run claude/codex to edit code
  3. evaluate    — docker build (cached), docker run with sandbox mounted, read reward
  4. _move_artifacts — sandbox + log moved into node directory

No Harbor required. Docker only for evaluation.
"""

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmarks.base import Benchmark
from benchmarks.utils import inject_skill, log_error, parse_run_cost, print_cost
from search.tree import TreeNode

AUTOLAB_DIR   = Path(__file__).parent          # benchmarks/autolab/
WORKSPACES    = AUTOLAB_DIR / "_workspaces"    # temp sandboxes + logs (cleaned up by _move_artifacts)
STAGED_SKILLS = AUTOLAB_DIR / "_staged_skills" # per-task skill staging between setup_node and run_agent

# Codex version used for AutoLab runs (independent of the globally installed codex).
# fire_bench keeps using the global binary (0.39.0).
CODEX_VERSION = "0.121.0"


class AutoLab(Benchmark):

    def __init__(self, benchmark_path: Path | None = None):
        self._root = Path(benchmark_path) if benchmark_path else AUTOLAB_DIR

    @property
    def name(self) -> str:
        return "autolab"

    @property
    def insights_dir(self) -> Path:
        return AUTOLAB_DIR / "insights"

    @property
    def log_root(self) -> Path:
        """
        run_search._move_artifacts uses this as the base for relative paths
        in packet["log_path"] and packet["repo_path"].
        It moves sandbox → node/sandbox and log → node/log.log.
        """
        return AUTOLAB_DIR

    # ── Instruction ──────────────────────────────────────────────────────────

    def get_instruction_path(self, task: str) -> Path:
        return self._root / "tasks" / task / "instruction.md"

    # ── Node setup ───────────────────────────────────────────────────────────

    def stage_skill(self, task: str, content: str) -> None:
        STAGED_SKILLS.mkdir(parents=True, exist_ok=True)
        (STAGED_SKILLS / f"{task}.md").write_text(content, encoding="utf-8")

    def prepend_skill_context(self, task: str, preamble: str) -> None:
        staged = STAGED_SKILLS / f"{task}.md"
        if staged.exists():
            staged.write_text(preamble + staged.read_text(encoding="utf-8"), encoding="utf-8")

    def setup_node(self, task: str, node: TreeNode) -> bool:
        """
        Validate skill.md exists and stage it for run_agent to inject into sandbox.
        Sandbox is created inside run_agent (we don't have the sandbox path here).
        """
        if not node.skill_path.exists():
            print(f"  [setup] ERROR: {node.skill_path} missing.")
            return False

        STAGED_SKILLS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(node.skill_path, STAGED_SKILLS / f"{task}.md")
        return True

    # ── Agent run ─────────────────────────────────────────────────────────────

    def run_agent(
        self, task: str, agent: str, model: str, run_env: dict,
        env_override: Path | None = None,
        node_dir: Path | None = None,
    ) -> str | None:
        """
        Copy environment/ (or env_override) to a fresh sandbox, inject skill,
        run claude/codex with cwd = sandbox so the agent edits files directly.
        Returns a timestamp_id run_id, or None on error.

        env_override lets a caller (e.g. vanilla-baseline loop) seed the
        sandbox from a previously-committed best state instead of the pristine
        environment/ directory.
        """
        import random
        WORKSPACES.mkdir(parents=True, exist_ok=True)
        timestamp = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(10000, 99999)}"

        # ── Sandbox ───────────────────────────────────────────────────────────
        env_src = env_override if env_override else self._root / "tasks" / task / "environment"
        if not env_src.exists():
            print(f"  [run] ERROR: environment not found: {env_src}")
            return None
        sandbox = WORKSPACES / f"{task}_{timestamp}"
        # When seeding from a committed best state, skip artefacts that are
        # per-iteration (reward output, injected skill files).
        shutil.copytree(
            str(env_src), str(sandbox),
            ignore=shutil.ignore_patterns("_reward", ".agents", "CLAUDE.md", "target"),
        )

        # ── Skill injection ───────────────────────────────────────────────────
        staged_skill = STAGED_SKILLS / f"{task}.md"
        skill_text = staged_skill.read_text(encoding="utf-8").strip() if staged_skill.exists() else ""

        # ── Instruction ───────────────────────────────────────────────────────
        instr_path = self._root / "tasks" / task / "instruction.md"
        instruction = instr_path.read_text(encoding="utf-8") if instr_path.exists() else task

        inject_skill(
            sandbox, agent, skill_text,
            skill_name="optimisation-strategy",
            skill_description=(
                "Apply when optimising code for performance. "
                "Always apply for any systems or algorithm optimisation task."
            ),
        )

        # ── Pre-build Docker image so the agent can use build_check.sh ─────────
        image_tag = f"autolab-{task.replace('_', '-')}"
        env_dir   = self._root / "tasks" / task / "environment"
        self._docker_build(env_dir, image_tag)

        # ── Write build_check.sh into sandbox ────────────────────────────────
        check_script = _make_build_check_script(
            env_dir, self._root / "tasks" / task / "tests", image_tag,
        )
        if check_script:
            p = sandbox / "build_check.sh"
            p.write_text(check_script, encoding="utf-8")
            p.chmod(0o755)

        # ── Build command ─────────────────────────────────────────────────────
        if agent in ("claude", "claude-code"):
            cmd = [
                "claude",
                "-p", instruction,
                "--model", model,
                "--output-format", "stream-json",
                "--verbose",
                "--add-dir", str(sandbox),
                "--dangerously-skip-permissions",
            ]
        elif agent == "codex":
            # Use npx to pin the version independently of the globally installed codex.
            # fire_bench uses the global codex (0.39.0); autolab uses a newer release.
            cmd = [
                "npx", "--yes", f"@openai/codex@{CODEX_VERSION}",
                "--model", model,
                "--config", "sandbox_mode=danger-full-access",
                "--ask-for-approval=never",
                "exec", "--skip-git-repo-check",
                instruction,
            ]
        else:
            print(f"  [run] Unknown agent: {agent!r} (choose 'claude' or 'codex')")
            return None

        # ── Run ───────────────────────────────────────────────────────────────
        log_file = WORKSPACES / f"log_{task}_{timestamp}.log"
        print(f"  [run] {agent}/{model} on {task}  sandbox={sandbox.name}")
        # Prevent the agent from pip-installing into the host conda env.
        agent_env = {**run_env, "PIP_REQUIRE_VIRTUALENV": "1"}
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"agent={agent}  model={model}  task={task}\n{'='*40}\n")
            f.flush()  # must flush before subprocess writes to the same fd
            subprocess.run(
                cmd, cwd=str(sandbox), env=agent_env,
                stdout=f, stderr=subprocess.STDOUT,
            )

        cost_info = parse_run_cost(log_file, agent, model)
        print_cost(cost_info, agent, model)

        return timestamp

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, task: str, run_id: str) -> dict:
        """
        Build the Docker image (cached), run tests/test.sh with sandbox mounted
        over /app, read /logs/verifier/reward.json.
        """
        sandbox  = WORKSPACES / f"{task}_{run_id}"
        log_file = WORKSPACES / f"log_{task}_{run_id}.log"
        task_dir = self._root / "tasks" / task

        # Relative paths from log_root (AUTOLAB_DIR) — used by _move_artifacts
        try:
            repo_rel = str(sandbox.relative_to(AUTOLAB_DIR))
            log_rel  = str(log_file.relative_to(AUTOLAB_DIR)) if log_file.exists() else ""
        except ValueError:
            repo_rel = str(sandbox)
            log_rel  = str(log_file)

        reward        = 0.0
        reward_detail: dict = {}

        if not sandbox.exists():
            print(f"  [eval] ERROR: sandbox not found: {sandbox}")
        else:
            image_tag = f"autolab-{task.replace('_', '-')}"
            if not self._docker_build(task_dir / "environment", image_tag):
                reward_detail = {"error": "docker_build_failed"}
            else:
                reward, reward_detail = self._docker_eval(
                    sandbox, task_dir / "tests" / "test.sh",
                    image_tag, task,
                )

        # Parse agent cost from log (log still in WORKSPACES before _move_artifacts)
        agent = run_id  # run_id is timestamp; agent type unknown here — use log metadata
        cost_info = _cost_from_log_header(log_file)

        print(f"  [eval] reward={reward:.4f}  task={task}")
        return {
            "task":          task,
            "run_id":        run_id,
            "log_path":      log_rel,
            "repo_path":     repo_rel,
            "score":         {"reward": reward},
            "reward_detail": reward_detail,
            "cost":          cost_info,
            "ran_at":        datetime.now().isoformat(),
        }

    def primary_score(self, packet: dict) -> float:
        return float(packet.get("score", {}).get("reward") or 0.0)

    # ── Docker helpers ────────────────────────────────────────────────────────

    def _docker_build(self, env_dir: Path, image_tag: str) -> bool:
        """Build the Docker image from environment/Dockerfile (uses layer cache)."""
        print(f"  [docker] Building {image_tag} ...")
        result = subprocess.run(
            ["docker", "build", "-t", image_tag, str(env_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  [docker] Build FAILED:\n{result.stderr[:1000]}")
            return False
        print(f"  [docker] Build OK: {image_tag}")
        return True

    def _docker_eval(
        self,
        sandbox: Path,
        test_sh: Path,
        image_tag: str,
        task: str,
    ) -> tuple[float, dict]:
        """
        Run tests/test.sh inside Docker with sandbox mounted at /app.
        Returns (reward_float, reward_detail_dict).

        Mount layout:
          {sandbox}  → /app            (agent's code, overrides image /app)
          {reward_dir} → /logs/verifier (output: reward.json, reward.txt)
          {test_sh}  → /eval.sh        (test script, read-only)

        Image retains: /tests/ (test data), /orig/ (hash), compiler toolchain.
        """
        reward_dir = sandbox / "_reward"
        reward_dir.mkdir(exist_ok=True)

        cpus, memory_mb, gpus = self._read_resource_limits(self._root / "tasks" / task)
        verifier_timeout = self._read_verifier_timeout(self._root / "tasks" / task)

        # Mount individual helper scripts from tests/ into /tests inside the image.
        # We cannot mount the whole tests/ directory because that would shadow
        # /tests/benchmark_matches.txt and /tests/benchmark_checksum.txt which
        # are generated during docker build from the baseline binary.
        extra_mounts: list[str] = []
        for script in sorted(test_sh.parent.glob("*.py")):
            extra_mounts += ["-v", f"{script}:/tests/{script.name}:ro"]

        gpu_flags = []
        if gpus > 0:
            device_ids = _pick_free_gpus(gpus)
            if device_ids:
                gpu_flags = ["--gpus", f"\"device={','.join(map(str, device_ids))}\""]
                print(f"  [docker] GPU(s) selected: {device_ids}")
            else:
                gpu_flags = ["--gpus", str(gpus)]
                print(f"  [docker] GPU selection fallback: --gpus {gpus}")

        cmd = [
            "docker", "run", "--rm",
            "--cpus",   str(cpus),
            "--memory", f"{memory_mb}m",
            *gpu_flags,
            "-v", f"{sandbox}:/app",
            "-v", f"{reward_dir}:/logs/verifier",
            "-v", f"{test_sh}:/eval.sh:ro",
            *extra_mounts,
            image_tag,
            "bash", "/eval.sh",
        ]
        print(f"  [docker] Running verifier (timeout={verifier_timeout}s) ...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=verifier_timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"  [docker] Verifier TIMEOUT after {verifier_timeout}s")
            return 0.0, {"error": "verifier_timeout"}

        # Persist verifier stdout/stderr so the reflector can see *why* the run
        # failed (build errors, correctness-diff output). Without this the
        # reflector only sees reward=0 / correctness=false and has to guess —
        # which was making tasks like concurrent_kv_wal fail indefinitely.
        combined = (
            (result.stdout or b"").decode("utf-8", errors="replace") +
            (result.stderr or b"").decode("utf-8", errors="replace")
        )
        if combined:
            print(combined[-3000:])
            (reward_dir / "verifier.log").write_text(combined, encoding="utf-8")
        if result.returncode not in (0, 1):  # test.sh exits 0 or 1 (set -e)
            print(f"  [docker] Verifier exited {result.returncode}")

        # Generate a compact diff between the pristine environment and the
        # agent's final sandbox. The reflector can read this (small) instead
        # of cat-ing every file in sandbox/ (thousands of tokens) to figure
        # out what the agent actually changed.
        # Wrapped in try/except — this is diagnostic only; a failure here must
        # never cause the already-computed reward to be lost.
        try:
            self._write_baseline_diff(task, sandbox, reward_dir)
        except Exception as e:
            log_error("_write_baseline_diff", e, task=task, sandbox=sandbox)

        reward_json = reward_dir / "reward.json"
        if not reward_json.exists():
            print(f"  [docker] WARNING: reward.json not produced")
            return 0.0, {
                "error": "no_reward_json",
                "verifier_output_tail": combined[-2000:] if combined else "",
            }

        try:
            detail = json.loads(reward_json.read_text(encoding="utf-8"))
            reward = float(detail.get("reward", 0.0))
            # Attach a tail of the verifier output so packet.json itself carries
            # diagnostic information (build errors or correctness diff) to the
            # reflector without requiring a separate file read.
            if combined:
                detail["verifier_output_tail"] = combined[-2000:]
            return reward, detail
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [docker] WARNING: bad reward.json ({e})")
            return 0.0, {
                "error": str(e),
                "verifier_output_tail": combined[-2000:] if combined else "",
            }

    def _write_baseline_diff(self, task: str, sandbox: Path, reward_dir: Path) -> None:
        """Dump a compact `diff -ruN baseline sandbox` into reward_dir.

        The reflector reads this instead of the full sandbox source tree, so
        that "what did this agent change?" costs ~100 tokens instead of the
        20k+ of cat-ing every *.go / *.py file.
        """
        baseline = self._root / "tasks" / task / "environment"
        if not baseline.exists():
            return
        try:
            result = subprocess.run(
                [
                    "diff", "-ruN",
                    "--exclude=_reward", "--exclude=.agents",
                    "--exclude=CLAUDE.md", "--exclude=.git",
                    "--exclude=__pycache__", "--exclude=node_modules",
                    str(baseline), str(sandbox),
                ],
                capture_output=True, timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            log_error("_write_baseline_diff", e, task=task, sandbox=sandbox)
            return
        # diff exits 0 if identical, 1 if different, >1 on error.
        # Either way, save whatever it produced. Decode with replace so binary
        # files (compiled objects, executables) don't crash the UTF-8 decode.
        out = (result.stdout or b"").decode("utf-8", errors="replace")
        # Cap diff at 200 KB — if the agent rewrote everything, a massive diff
        # is less useful than a truncated one; better to say "too big, see
        # sandbox/" than to dump megabytes.
        MAX_DIFF_BYTES = 200_000
        if len(out.encode("utf-8", errors="ignore")) > MAX_DIFF_BYTES:
            out = (out[:MAX_DIFF_BYTES]
                   + "\n\n[...diff truncated — exceeded 200 KB, inspect sandbox/ directly for full details...]\n")
        (reward_dir / "diff_from_baseline.log").write_text(out, encoding="utf-8")


    def _read_resource_limits(self, task_dir: Path) -> tuple[int, int, int]:
        """Return (cpus, memory_mb, gpus) from task.toml, or defaults."""
        toml_path = task_dir / "task.toml"
        if not toml_path.exists():
            return 4, 3072, 0
        try:
            import tomllib
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            env = data.get("environment", {})
            return (int(env.get("cpus", 4)),
                    int(env.get("memory_mb", 3072)),
                    int(env.get("gpus", 0)))
        except Exception as e:
            log_error("_read_resource_limits", e, toml_path=toml_path)
            return 4, 3072, 0

    def _read_verifier_timeout(self, task_dir: Path) -> int:
        """Return verifier timeout in seconds from task.toml, or default 900."""
        toml_path = task_dir / "task.toml"
        if not toml_path.exists():
            return 900
        try:
            import tomllib
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            return int(data.get("verifier", {}).get("timeout_sec", 900))
        except Exception as e:
            log_error("_read_verifier_timeout", e, toml_path=toml_path)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_build_check_script(env_dir: Path, tests_dir: Path, image_tag: str) -> str:
    """Return the content of build_check.sh for the sandbox, or '' if not applicable.

    Strategy per language:
      C/C++  — gcc/make are on the host, so compile with `make` directly, then
               run verify_correctness.py (also on host) for a correctness check.
      Go     — not on host; use `docker run` to compile inside the image.
      Rust   — not on host; use `docker run` to compile inside the image.
    """
    dockerfile = env_dir / "Dockerfile"
    if not dockerfile.exists():
        return ""
    text = dockerfile.read_text(encoding="utf-8", errors="ignore").lower()

    verify_py = tests_dir / "verify_correctness.py"

    if "golang:" in text or "from golang" in text:
        return (
            "#!/usr/bin/env bash\n"
            "# Verify your Go changes compile. Run before finishing.\n"
            "set -e\n"
            f"docker run --rm -v \"$(pwd):/app\" {image_tag} go build ./...\n"
            "echo 'Build OK'\n"
        )

    if "rust:" in text or "from rust" in text:
        return (
            "#!/usr/bin/env bash\n"
            "# Verify your Rust changes compile. Run before finishing.\n"
            "set -e\n"
            f"docker run --rm -v \"$(pwd):/app\" -w /app {image_tag} cargo build 2>&1\n"
            "echo 'Build OK'\n"
        )

    if "gcc" in text or "g++" in text or (env_dir / "Makefile").exists():
        return (
            "#!/usr/bin/env bash\n"
            "# Verify your changes compile. Run before finishing.\n"
            "set -e\n"
            "make 2>&1\n"
            "echo 'Compile OK'\n"
        )

    return ""


def _pick_free_gpus(n: int) -> list[int]:
    """Return the indices of the n GPUs with the most free memory.

    Uses `nvidia-smi` to query free memory per device and sorts by descending
    free memory. Falls back to an empty list if nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus: list[tuple[int, int]] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                gpus.append((int(parts[0].strip()), int(parts[1].strip())))
        gpus.sort(key=lambda x: x[1], reverse=True)  # most free first
        return [idx for idx, _ in gpus[:n]]
    except Exception:
        return []


def _cost_from_log_header(log_file: Path) -> dict:
    """
    Find agent/model from the log written by run_agent and return parse_run_cost().

    Two formats handled:
      1. Our header (written after f.flush() fix): "agent=X  model=Y  task=Z"
      2. Codex 0.121.0 banner (pre-flush legacy): "OpenAI Codex ..." + "model: Y"
    """
    if not log_file.exists():
        return {}
    try:
        import re as _re
        text  = log_file.read_text(encoding="utf-8", errors="ignore")
        lines = text.split("\n")[:20]
        agent = model = None

        # Format 1: our header line "agent=codex  model=gpt-5  task=..."
        for line in lines:
            am = _re.search(r"agent=(\S+)", line)
            mm = _re.search(r"model=(\S+)", line)
            if am and mm:
                agent = am.group(1)
                model = mm.group(1)
                break

        # Format 2: codex banner (header lost due to unflushed buffer)
        if not agent:
            for line in lines:
                if "OpenAI Codex" in line or "codex" in line.lower():
                    agent = "codex"
                    break
            for line in lines:
                mm = _re.match(r"model:\s*(\S+)", line.strip())
                if mm:
                    model = mm.group(1)
                    break

        if not agent or not model:
            return {}
        from benchmarks.utils import parse_run_cost
        return parse_run_cost(log_file, agent, model)
    except Exception as e:
        log_error("_cost_from_log_header", e, log_file=log_file)
