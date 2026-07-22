"""
benchmarks/utils.py — Shared helpers for benchmark agent runners.
"""

import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path

ERROR_LOG = Path(__file__).parent.parent / "errors.log"

# `codex exec` reports only ONE cumulative "tokens used" number with no
# input/output/reasoning split, so we cannot price output/reasoning tokens
# (billed ~4x input) accurately. This calibration multiplier scales the codex
# cost estimate up to track real spend (observed ~3-4x underestimate). Bias
# high on purpose: the estimate feeds the --max_cost_usd hard cap, so
# overestimating stops runs early rather than overspending. Tune via env.
CODEX_COST_MULT = float(os.environ.get("METASCI_CODEX_COST_MULT", "3.5"))


def log_error(context: str, exc: BaseException, **extra) -> None:
    """Append a full traceback + context to errors.log and re-raise."""
    msg = (
        f"\n{'='*60}\n"
        f"[{datetime.now().isoformat()}] ERROR in {context}\n"
        + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    )
    if extra:
        msg += "Context:\n" + "".join(f"  {k}: {v!r}\n" for k, v in extra.items())
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(msg)
    except Exception:
        pass  # don't recurse
    print(f"  [ERROR] {context}: {exc}  (see errors.log)")
    raise

# ---------------------------------------------------------------------------
# Pricing tables  ($ per million tokens, as of mid-2025)
# ---------------------------------------------------------------------------

# Claude: (input, output, cache_write, cache_read)
_CLAUDE_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-4-6":       (15.00, 75.00, 18.75, 1.50),
    "claude-opus-4-5":       (15.00, 75.00, 18.75, 1.50),
    "claude-sonnet-4-6":     ( 3.00, 15.00,  3.75, 0.30),
    "claude-sonnet-4-5":     ( 3.00, 15.00,  3.75, 0.30),
    "claude-3-7-sonnet":     ( 3.00, 15.00,  3.75, 0.30),
    "claude-3-5-sonnet":     ( 3.00, 15.00,  3.75, 0.30),
    "claude-haiku-4-5":      ( 0.80,  4.00,  1.00, 0.08),
    "claude-3-5-haiku":      ( 0.80,  4.00,  1.00, 0.08),
    "claude-3-haiku":        ( 0.25,  1.25,  0.30, 0.03),
}

# OpenAI/codex: (input, output)
_OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5":       (10.00, 40.00),   # estimate
    "o4-mini":     ( 1.10,  4.40),
    "o3":          (10.00, 40.00),
    "o3-mini":     ( 1.10,  4.40),
    "o1":          (15.00, 60.00),
    "o1-mini":     ( 3.00, 12.00),
    "gpt-4o":      ( 2.50, 10.00),
    "gpt-4o-mini": ( 0.15,  0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4":       (30.00, 60.00),
}


def _claude_unit_price(model: str) -> tuple[float, float, float, float]:
    for key, prices in _CLAUDE_PRICING.items():
        if key in model or model.startswith(key):
            return prices
    return (3.00, 15.00, 3.75, 0.30)   # sonnet default


def _openai_unit_price(model: str) -> tuple[float, float]:
    m = model.lower()
    for key, prices in _OPENAI_PRICING.items():
        if key in m:
            return prices
    return (10.00, 40.00)   # conservative default


# ---------------------------------------------------------------------------
# Cost parsing
# ---------------------------------------------------------------------------

def parse_run_cost(log_file: Path, agent: str, model: str) -> dict:
    """
    Parse an agent run log for cost / token usage.

    Returns a dict with some subset of:
      cost_usd         float  — total cost in USD
      input_tokens     int
      output_tokens    int
      cache_read_tokens  int   (Claude only)
      cache_write_tokens int   (Claude only)
      cost_estimated   bool   — True if cost was computed from pricing table,
                                False/absent if reported directly by the agent CLI

    Returns {} if no useful information was found.
    """
    if not log_file.exists():
        return {}
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    if agent in ("claude", "claude-code"):
        return _parse_claude_cost(text, model)
    elif agent == "codex":
        return _parse_codex_cost(text, model)
    return {}


def _parse_claude_cost(text: str, model: str) -> dict:
    """
    Claude Code --output-format stream-json emits one JSON object per line.
    The final result object has total_cost_usd and authoritative usage totals.
    Intermediate assistant-type objects repeat the same message ID many times
    (streaming chunks), so accumulating them overcounts tokens.
    Strategy: prefer the result object's usage; fall back to per-message accumulation
    with deduplication by message ID.
    """
    cost_usd     = None
    result_usage: dict = {}
    seen_msg_ids: set  = set()
    input_tok    = 0
    output_tok   = 0
    cache_read   = 0
    cache_write  = 0

    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Direct cost field (present in result objects)
        for key in ("cost_usd", "total_cost_usd"):
            if key in obj:
                try:
                    cost_usd = float(obj[key])
                except (TypeError, ValueError):
                    pass

        obj_type = obj.get("type", "")

        # result object has authoritative session-level usage totals
        if obj_type == "result" and "usage" in obj:
            result_usage = obj["usage"]
            continue

        # assistant-type objects: deduplicate by message ID to avoid streaming repetition
        if obj_type == "assistant":
            msg = obj.get("message") or {}
            msg_id = msg.get("id", "")
            if msg_id and msg_id in seen_msg_ids:
                continue
            if msg_id:
                seen_msg_ids.add(msg_id)
            usage = msg.get("usage") or {}
        else:
            usage = obj.get("usage") or {}

        input_tok   += int(usage.get("input_tokens", 0))
        output_tok  += int(usage.get("output_tokens", 0))
        cache_read  += int(usage.get("cache_read_input_tokens", 0))
        cache_write += int(usage.get("cache_creation_input_tokens", 0))

    # Prefer result object's usage if present (authoritative)
    if result_usage:
        input_tok   = int(result_usage.get("input_tokens", input_tok))
        output_tok  = int(result_usage.get("output_tokens", output_tok))
        cache_read  = int(result_usage.get("cache_read_input_tokens", cache_read))
        cache_write = int(result_usage.get("cache_creation_input_tokens", cache_write))

    if not input_tok and not output_tok and cost_usd is None:
        return {}

    result: dict = {
        "input_tokens":       input_tok,
        "output_tokens":      output_tok,
        "cache_read_tokens":  cache_read,
        "cache_write_tokens": cache_write,
    }

    if cost_usd is not None:
        result["cost_usd"] = round(cost_usd, 6)
    else:
        # Compute from pricing table
        in_p, out_p, cw_p, cr_p = _claude_unit_price(model)
        est = (input_tok * in_p + output_tok * out_p
               + cache_write * cw_p + cache_read * cr_p) / 1_000_000
        result["cost_usd"]       = round(est, 6)
        result["cost_estimated"] = True

    return result


def _parse_codex_cost(text: str, model: str) -> dict:
    """
    Codex outputs usage info in JSON lines or plain-text summaries.
    Parse prompt/completion token counts and compute cost from pricing table.

    Supported formats:
      - JSON lines with {"usage": {"prompt_tokens": N, "completion_tokens": N}}
      - Plain-text "tokens used\\n69,752"  (codex 0.39.0 / fire_bench format)
      - Plain-text "prompt_tokens: N" / "completion_tokens: N"
    """
    prompt_tok     = 0
    completion_tok = 0

    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = obj.get("usage") or {}
        prompt_tok     += int(usage.get("prompt_tokens",     0) or usage.get("input_tokens",  0))
        completion_tok += int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0))

    # Fallback: regex on plain-text summary lines.
    # Require ':' or '=' after the key and capture digits/commas only — a loose
    # `[^\d]+` was matching narrative prose (e.g. "prompt tokens from loss.\n- 4"
    # captured `4` from "4-bit base model") and short-circuited the reliable
    # "tokens used" fallback below.
    if not prompt_tok:
        m = re.search(r"prompt[_\s]tokens\s*[:=]\s*([\d,]+)", text, re.IGNORECASE)
        if m:
            prompt_tok = int(m.group(1).replace(",", ""))
    if not completion_tok:
        m = re.search(r"completion[_\s]tokens\s*[:=]\s*([\d,]+)", text, re.IGNORECASE)
        if m:
            completion_tok = int(m.group(1).replace(",", ""))

    # Fallback: codex 0.39.0 "[timestamp] tokens used: 17,819" — take last (cumulative total)
    if not prompt_tok and not completion_tok:
        matches = re.findall(r"tokens\s+used[:\s]+([\d,]+)", text, re.IGNORECASE)
        if matches:
            prompt_tok = int(matches[-1].replace(",", ""))

    if not prompt_tok and not completion_tok:
        return {}

    in_p, out_p = _openai_unit_price(model)
    # Codex gives no output/reasoning split -> scale up by CODEX_COST_MULT so the
    # estimate (and the --max_cost_usd cap it feeds) tracks real spend.
    cost = (prompt_tok * in_p + completion_tok * out_p) / 1_000_000 * CODEX_COST_MULT

    return {
        "input_tokens":  prompt_tok,
        "output_tokens": completion_tok,
        "cost_usd":      round(cost, 6),
        "cost_estimated": True,
        "cost_mult":     CODEX_COST_MULT,
    }


def print_cost(cost_info: dict, agent: str, model: str) -> None:
    """Pretty-print cost info after an agent run."""
    if not cost_info:
        return
    cost = cost_info.get("cost_usd")
    est  = " (est.)" if cost_info.get("cost_estimated") else ""
    inp  = cost_info.get("input_tokens", 0)
    out  = cost_info.get("output_tokens", 0)
    cr   = cost_info.get("cache_read_tokens", 0)
    cw   = cost_info.get("cache_write_tokens", 0)

    cost_str = f"${cost:.4f}{est}" if cost is not None else "unknown"
    tok_str  = f"in={inp:,}  out={out:,}"
    if cr or cw:
        tok_str += f"  cache_read={cr:,}  cache_write={cw:,}"
    print(f"  [cost] {agent}/{model}: {cost_str}  {tok_str}")


COST_LEDGER_HEADER = "ran_at\trole\tmodel\tcost_usd\tinput_tokens\toutput_tokens\tcache_read_tokens\tcache_write_tokens\tnode\n"


def append_cost_ledger(
    ledger_path: Path,
    cost_info: dict,
    role: str,
    model: str,
    node_name: str = "",
) -> None:
    """Append one cost record to a costs.tsv ledger. No-op if cost_info is empty."""
    if not cost_info:
        return
    if not ledger_path.exists():
        ledger_path.write_text(COST_LEDGER_HEADER, encoding="utf-8")
    row = "\t".join([
        datetime.now().isoformat(),
        role,
        model,
        f"{cost_info.get('cost_usd', 0.0):.6f}",
        str(cost_info.get("input_tokens", 0)),
        str(cost_info.get("output_tokens", 0)),
        str(cost_info.get("cache_read_tokens", 0)),
        str(cost_info.get("cache_write_tokens", 0)),
        node_name,
    ])
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(row + "\n")


def write_claude_skill(sandbox: Path, body: str) -> None:
    """
    Write skill as CLAUDE.md in sandbox.
    Claude Code auto-loads CLAUDE.md from cwd.
    """
    (sandbox / "CLAUDE.md").write_text(
        "# Strategy Guidance (from self-improvement reflector)\n\n"
        + body + "\n",
        encoding="utf-8",
    )


def write_codex_skill(
    sandbox: Path,
    body: str,
    skill_name: str = "agent-skill",
    description: str = "Apply this skill for the current task.",
) -> None:
    """
    Write skill into Codex's native .agents/skills/ structure.
    Codex auto-discovers .agents/skills/{name}/SKILL.md in cwd and loads it
    when the description matches the task context.

    Layout written:
      .agents/skills/{skill_name}/SKILL.md          — skill body with frontmatter
      .agents/skills/{skill_name}/agents/openai.yaml — display config
    """
    skill_dir = sandbox / ".agents" / "skills" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = (
        "---\n"
        f"name: {skill_name}\n"
        f'description: "{description}"\n'
        "---\n\n"
        + body
    )
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    agents_dir = skill_dir / "agents"
    agents_dir.mkdir(exist_ok=True)
    (agents_dir / "openai.yaml").write_text(
        "interface:\n"
        "  display_name: Agent Skill\n"
        "  short_description: Strategy guidance from self-improvement reflector\n"
        "policy:\n"
        "  allow_implicit_invocation: true\n",
        encoding="utf-8",
    )


def inject_skill(
    sandbox: Path,
    agent: str,
    skill_text: str,
    skill_name: str = "agent-skill",
    skill_description: str = "Apply this skill for the current task.",
) -> None:
    """
    Inject skill into sandbox for the given agent type:
      - claude / claude-code → CLAUDE.md (auto-loaded from cwd)
      - codex               → .agents/skills/{skill_name}/SKILL.md
    No-op if skill_text is empty.
    """
    if not skill_text:
        return
    if agent in ("claude", "claude-code"):
        write_claude_skill(sandbox, skill_text)
    elif agent == "codex":
        write_codex_skill(sandbox, skill_text, skill_name, skill_description)
