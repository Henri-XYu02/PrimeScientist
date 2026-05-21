"""
benchmarks/base.py — Abstract benchmark interface.

A Benchmark wraps everything benchmark-specific:
  - where to find the instruction for a task
  - how to copy the node's skill snapshot so the agent reads it
  - how to run the agent on a task
  - how to evaluate the result and produce a packet dict
  - how to extract the primary scalar score for PUCT

To add a new benchmark, subclass Benchmark and implement all abstract methods,
then register it in benchmarks/registry.py.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from search.tree import TreeNode


class Benchmark(ABC):

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'fire_bench' or 'autolab'."""

    @property
    @abstractmethod
    def insights_dir(self) -> Path:
        """
        Directory where cross-task {task}_insight.md files are stored.
        The reflector reads these when initialising a new task; agents load
        them as context for related tasks.
        """

    # ── Task interface ────────────────────────────────────────────────────────

    @abstractmethod
    def get_instruction_path(self, task: str) -> Path:
        """
        Path to the instruction file the reflector should read when
        initialising a new task (e.g. instruction.txt / instruction.md).
        """

    @abstractmethod
    def setup_node(self, task: str, node: TreeNode) -> bool:
        """
        Copy / inject the node's skill snapshot so the agent will read it.
        Called immediately before run_agent().
        Returns False if setup fails (missing files, etc.).
        """

    @abstractmethod
    def run_agent(
        self,
        task: str,
        agent: str,
        model: str,
        run_env: dict,
        env_override: "Path | None" = None,
        node_dir: "Path | None" = None,
    ) -> str | None:
        """
        Run the agent on the task.
        Returns an opaque run_id used by evaluate(), or None on failure.

        env_override: if set, seed the agent's workspace from this path
                      (parent node's sandbox) instead of the pristine baseline.
        node_dir:     this node's directory in the search tree. Benchmarks
                      may copy proposal-specific artifacts (e.g. diff.py,
                      changes.md) from here into the agent's sandbox.
        """

    @abstractmethod
    def evaluate(self, task: str, run_id: str) -> dict:
        """
        Evaluate the agent's run.
        Returns a packet dict that must contain at minimum:
          {"score": {<metric_name>: <float>, ...}, "ran_at": "<iso timestamp>"}
        May also include benchmark-specific fields (log_path, conclusion, etc.)
        that the reflector can read from the search tree.
        """

    @abstractmethod
    def primary_score(self, packet: dict) -> float:
        """
        Extract a single scalar (higher = better) from a packet for use
        in PUCT Q-value backpropagation.
        """

    # ── Optional hooks / properties ───────────────────────────────────────────

    @property
    def log_root(self) -> "Path | None":
        """
        Root directory under which run logs and sandboxes are stored.
        Used by run_search to move artifacts into the node directory.
        Return None if not applicable (artifacts are already in the right place).
        """
        return None

    def post_init_node(self, task: str, node: "TreeNode") -> None:
        """
        Called after the reflector initialises a new root node.
        Use to copy skill/insight snapshots to any live benchmark locations
        that the agent runner reads from.
        Default: no-op (override only if needed).
        """

    def prepend_skill_context(self, task: str, preamble: str) -> None:
        """
        Prepend preamble text to the staged skill file so the agent reads it.
        Called by run_search after setup_node when code inheritance is active,
        so the agent knows the workspace is not the pristine baseline.
        Default: no-op. Override in benchmarks that stage a skill file.
        """

    def stage_skill(self, task: str, content: str) -> None:
        """
        Overwrite the live skill file the agent reads with `content`. Used by
        the vanilla baseline to inject prior-iteration history as the skill,
        so the agent sees what's already been tried.
        Default: no-op. Override in benchmarks that stage a skill file.
        """

    @property
    def agent_inheritance_preamble(self) -> str:
        """
        Preamble prepended to the staged skill when the child inherits the
        parent's sandbox. Use {parent_q} placeholder for the parent's Q score.
        Return "" to skip the preamble entirely.

        Default style: incremental-change ("delta") workflow — suitable for
        code-optimisation benchmarks (autolab) where the skill is a high-level
        pointer and the code is the source of truth. Override for benchmarks
        that need a complete plan in skill.md.
        """
        return (
            "> **INHERITED WORKSPACE** (score={parent_q:.3f}): This sandbox is "
            "seeded from the parent node's best implementation — the code here "
            "already includes the parent's changes. Do NOT re-implement what is "
            "already present. Apply only the ADDITIONAL changes described "
            "below.\n\n"
        )

    @property
    def proposal_block(self) -> str:
        """
        Reflector instructions for creating child proposal directories.
        Empty string → reflect_and_propose uses DEFAULT_PROPOSAL_BLOCK
        (direct skill.md + prior.json write, suitable for autolab).
        Override to return CODEACT_PROPOSAL_BLOCK or a custom block (e.g.
        fire_bench, where each child needs a complete experimental plan
        produced by a transformation script).
        """
        return ""

    @property
    def reflect_template(self) -> str:
        """
        Full REFLECT_PROMPT template used by reflect_and_propose.
        Empty string → uses the default REFLECT_PROMPT from search.reflector
        (references sandbox/_reward/diff_from_baseline.log and verifier.log,
        suitable for autolab). Override for benchmarks whose sandboxes have a
        different layout (e.g. fire_bench uses FIRE_BENCH_REFLECT_PROMPT).
        """
        return ""

    @property
    def reflector_inheritance_note(self) -> str:
        """
        Note appended to the reflector prompt when the parent node has a
        sandbox that children will inherit. Tells the reflector how to
        structure proposal skill.md files. Use {parent_q} placeholder.
        Return "" to skip the note entirely.

        Default style: instruct delta-only proposals (autolab-style).
        Override for benchmarks where each child needs a self-contained plan.
        """
        return (
            "\n\nNOTE — code inheritance active: child agents will be seeded "
            "from this node's `sandbox/` directory (parent Q={parent_q:.3f}), "
            "NOT the pristine baseline. Write each child's skill.md as "
            "ADDITIONAL incremental changes to apply on top of what this node "
            "already implemented. Describe only the delta — do not re-describe "
            "the full implementation from scratch.\n"
        )
