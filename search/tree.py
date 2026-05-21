"""
search/tree.py — Filesystem-backed MCTS search tree for FIRE-Bench skill search.

Directory layout for each node:
  {node_dir}/
    skill.md              ← task-specific experimental plan snapshot
    {task}_insight.md     ← consolidated insight file snapshot
    packet.json           ← {log_path, score, agent_conclusion, fp/fn ...} (empty until run)
    stats.json            ← {visits, Q, P}
    prior.json            ← {estimate, rationale} (absent in root)
    children/             ← child nodes (proposals created by reflector)

BAVT selection: sample child proportional to weight^alpha
  weight = Q               for visited children (actual measured score)
         = parent.Q × √P   for unvisited children (inherit parent quality,
                           softened by reflector's prior — √ because P is
                           empirically noisy)
  alpha  = 1/r_t           (r_t = remaining budget ratio; high budget → alpha≈1
                            explore, low → exploit)
"""

import json
import math
import random
from pathlib import Path

STATS_FILE  = "stats.json"
PACKET_FILE = "packet.json"
PRIOR_FILE  = "prior.json"
CHILDREN_DIR = "children"

DEFAULT_STATS = {"visits": 0, "Q": 0.0, "P": 1.0}


class TreeNode:
    def __init__(self, path: Path, task: str):
        self.path = path
        self.task = task
        path.mkdir(parents=True, exist_ok=True)
        (path / CHILDREN_DIR).mkdir(exist_ok=True)
        if not (path / STATS_FILE).exists():
            self._write_stats(dict(DEFAULT_STATS))

    # ── File paths ──────────────────────────────────────────────────────────

    @property
    def skill_path(self) -> Path:
        return self.path / "skill.md"

    @property
    def insight_path(self) -> Path:
        return self.path / f"{self.task}_insight.md"

    @property
    def children_dir(self) -> Path:
        return self.path / CHILDREN_DIR

    # ── Stats ────────────────────────────────────────────────────────────────

    def _read_stats(self) -> dict:
        p = self.path / STATS_FILE
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else dict(DEFAULT_STATS)

    def _write_stats(self, stats: dict) -> None:
        (self.path / STATS_FILE).write_text(json.dumps(stats, indent=2), encoding="utf-8")

    @property
    def visits(self) -> int:
        return self._read_stats().get("visits", 0)

    @property
    def Q(self) -> float:
        return self._read_stats().get("Q", 0.0)

    @property
    def P(self) -> float:
        return self._read_stats().get("P", 1.0)

    def set_prior(self, p: float) -> None:
        stats = self._read_stats()
        stats["P"] = max(0.0, min(1.0, p))
        self._write_stats(stats)

    def record_visit(self, f1: float) -> None:
        """Increment visits and update Q as running average."""
        stats = self._read_stats()
        n = stats.get("visits", 0)
        q = stats.get("Q", 0.0)
        stats["visits"] = n + 1
        stats["Q"] = (q * n + f1) / (n + 1)
        self._write_stats(stats)

    # ── Packet ───────────────────────────────────────────────────────────────

    def has_packet(self) -> bool:
        p = self.path / PACKET_FILE
        return p.exists() and p.stat().st_size > 5

    def read_packet(self) -> dict:
        p = self.path / PACKET_FILE
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def write_packet(self, packet: dict) -> None:
        (self.path / PACKET_FILE).write_text(json.dumps(packet, indent=2), encoding="utf-8")

    # ── Children ─────────────────────────────────────────────────────────────

    def get_children(self) -> list["TreeNode"]:
        result = []
        if not self.children_dir.exists():
            return result
        for d in sorted(self.children_dir.iterdir()):
            # Exclude hidden dirs: .archived_* (bad skill.md) and .pruned_* (low score)
            if d.is_dir() and not d.name.startswith(".") and (d / STATS_FILE).exists():
                result.append(TreeNode(d, self.task))
        return result

    def pruned_child_count(self) -> int:
        """Number of score-pruned child dirs (.pruned_*)."""
        if not self.children_dir.exists():
            return 0
        return sum(
            1 for d in self.children_dir.iterdir()
            if d.is_dir() and d.name.startswith(".pruned_")
        )

    def add_child(self, name: str) -> "TreeNode":
        return TreeNode(self.children_dir / name, self.task)

    def is_expanded(self) -> bool:
        return bool(self.get_children())

    # ── Parent tracking (via directory structure) ─────────────────────────────

    def parent(self) -> "TreeNode | None":
        # node path: .../root/children/proposal_0
        # parent:    .../root
        candidate = self.path.parent.parent
        if (candidate / STATS_FILE).exists():
            return TreeNode(candidate, self.task)
        return None

    # ── Backpropagation ───────────────────────────────────────────────────────

    def backpropagate(self, f1: float) -> None:
        """Walk from this node to root, updating Q and visits."""
        node: TreeNode | None = self
        while node is not None:
            node.record_visit(f1)
            node = node.parent()

    # ── Repr ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"TreeNode({self.path.name}, Q={self.Q:.2f}, visits={self.visits}, P={self.P:.2f})"


# ---------------------------------------------------------------------------
# BAVT traversal
# ---------------------------------------------------------------------------

def _bavt_weight(node: TreeNode, alpha: float, parent: "TreeNode | None" = None) -> float:
    """BAVT node weight.

    Visited   → Q^alpha                  (actual measured performance)
    Unvisited → (parent.Q × √P)^alpha    (parent's proven quality scaled by
                                           a softened reflector prior √P)

    alpha = 1/r_t (remaining budget ratio):
      alpha=1  (full budget)  → weight ≈ virtual_Q   — broad exploration
      alpha→∞  (budget gone)  → mass concentrates on max — pure exploitation

    √P (vs P) is used because empirically the reflector's prior is noisy and
    hedged (corr(P, Q) ≈ 0.18 on prior runs); √ compresses the range so P
    acts as a soft tiebreaker rather than a strong selector. Unvisited with
    no parent info or parent.Q=0 falls back to √P^alpha. A small epsilon
    prevents zero-weight dead ends.
    """
    if node.visits > 0:
        q = node.Q
        return q ** alpha if q > 0.0 else 1e-6
    # Unvisited: inherit parent's proven quality, softened by √P prior
    sqrt_p = math.sqrt(max(0.0, node.P))
    if parent is not None and parent.Q > 0.0:
        virtual_q = parent.Q * sqrt_p
    else:
        virtual_q = sqrt_p
    return virtual_q ** alpha if virtual_q > 0.0 else 1e-6



def node_depth(node: TreeNode) -> int:
    """Count how many 'children' directories are in the path (= tree depth)."""
    return str(node.path).count("/children/")


def _subtree_exhausted(node: "TreeNode") -> bool:
    """
    Returns True if every node in this subtree has been run (no unrun leaves).
    Used to gate adding more proposals to a parent: we only branch wider once
    the entire existing subtree has been fully explored.
    """
    if not node.has_packet():
        return False  # this node itself hasn't been run yet
    return all(_subtree_exhausted(ch) for ch in node.get_children())


def puct_select_leaf(
    root: TreeNode,
    alpha: float = 1.0,
    max_depth: int = 0,
    max_proposals: int = 0,
) -> TreeNode:
    """
    Traverse tree using BAVT stochastic node selection.

    At each level, samples a child proportional to Q^alpha:
      alpha=1  (full budget)  → weights ≈ Q, broad exploration
      alpha→∞  (budget gone)  → mass concentrates on highest-Q child

    Stops at:
      - An unrun node (has_packet=False)                       → run it
      - A run node with no children                            → initial expansion
      - A run node where ALL children are evaluated AND under
        the cap (max_proposals > 0)                            → add more proposals
      - A node at max_depth (if max_depth > 0)                 → unexpandable leaf

    Critically: if a node has unrun children we descend into them first —
    we never ask for more proposals until every current child has a result.
    """
    node = root
    while node.has_packet():
        if max_depth > 0 and node_depth(node) >= max_depth:
            return node
        children = node.get_children()
        if not children:
            return node  # needs initial expansion
        if max_proposals > 0 and len(children) < max_proposals and all(_subtree_exhausted(ch) for ch in children):
            return node
        weights = [_bavt_weight(ch, alpha, parent=node) for ch in children]
        node = random.choices(children, weights=weights, k=1)[0]
    return node  # unrun → run it


def best_node(root: TreeNode) -> tuple[float, TreeNode]:
    """Return (best_Q, node) across the entire tree."""
    best_q, best = root.Q, root
    for child in root.get_children():
        q, n = best_node(child)
        if q > best_q:
            best_q, best = q, n
    return best_q, best


# ---------------------------------------------------------------------------
# Tree summary (for reflector context)
# ---------------------------------------------------------------------------

def tree_summary(root: TreeNode, indent: int = 0, max_depth: int = 6) -> str:
    """Compact text representation of the tree.

    max_depth is interpreted as the deepest indent level rendered:
      max_depth=0 → only the root node
      max_depth=1 → root + direct children (2 levels)
      max_depth=2 → root + children + grandchildren (3 levels)
      ...
    """
    if indent > max_depth:
        return ""
    prefix = "  " * indent
    pkt = root.read_packet()
    if root.has_packet():
        sc   = pkt.get("score", {})
        f1   = sc.get("f1", "?")
        conc = pkt.get("agent_conclusion", "")[:80]
        status = f"f1={f1}  '{conc}'"
    else:
        status = "not run"

    # Read prior if exists
    prior_file = root.path / PRIOR_FILE
    prior_str  = ""
    if prior_file.exists():
        try:
            pr = json.loads(prior_file.read_text(encoding="utf-8"))
            rat = pr.get("rationale", "")[:60]
            prior_str = f"  prior={pr.get('estimate', '?')}  rationale='{rat}'"
        except (json.JSONDecodeError, ValueError):
            pass

    line = f"{prefix}[{root.path.name}]  Q={root.Q:.2f}  visits={root.visits}{prior_str}  {status}"
    lines = [line]
    for child in root.get_children():
        lines.append(tree_summary(child, indent + 1, max_depth))
    return "\n".join(filter(None, lines))


def load_tree(tree_dir: Path, task: str) -> TreeNode | None:
    """Load existing tree root from disk, or None if not initialised."""
    root_path = tree_dir / "root"
    if root_path.exists() and (root_path / STATS_FILE).exists():
        return TreeNode(root_path, task)
    return None


def init_tree(tree_dir: Path, task: str) -> TreeNode:
    """Create a fresh root node (no skill files yet — reflector writes them)."""
    root_path = tree_dir / "root"
    return TreeNode(root_path, task)
