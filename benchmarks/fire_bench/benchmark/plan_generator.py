import json
import argparse
import os
from pathlib import Path
from typing import Optional

import fitz
import openai
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# The LLM already knows: research question, models, datasets, minimal hints
# (all from instruction.txt). We ask it ONLY for what's missing there.
SUPPLEMENTARY_PLAN_PROMPT = """\
You are an expert research engineer specializing in reproducing machine learning
and NLP experiments from academic papers.

A coding agent has already been given a base instruction (shown below under
"BASE INSTRUCTION") that tells it the research question, available models,
available datasets, and a few high-level hints.

Your job is to read the full paper and write a **supplementary experiment plan**
that gives the agent the additional detail it needs to run the experiments
faithfully — **without repeating anything already stated in the base instruction**.

Specifically, generate a Markdown document with ONLY the following sections
(omit any section that has nothing new to add beyond the base instruction):

## Experimental Design
For each distinct experiment in the paper:
- Numbered task with a short name
- Precise procedure (input format, prompting strategy, number of samples,
  any shuffling / ordering of conditions, splits to use)
- What to record (raw outputs, extracted answers, intermediate statistics)
- Correspondence to a Figure / Table / Section in the paper

## Evaluation Details
- Exact metric definitions and computation (formulas, library calls, edge cases)
- How to normalize or pre-process model outputs before scoring
- Aggregation strategy (per-sample → per-condition → overall)

## Implementation Notes
- Key pitfalls or caveats explicitly mentioned in the paper
- Any special prompt templates, delimiters, or output-parsing logic required
- Order of operations that must be respected (e.g., "evaluate without hints first")

---

Rules:
1. Do NOT restate the research question, model names, dataset names, or any
   content already present in the base instruction. Do NOT expose the conclusions in the paper.
2. Be concrete: use exact numbers, thresholds, prompt wording, and metric
   definitions from the paper.  Mark missing values as [uncertain].
3. Be complete: cover every experiment that maps to a figure, table, or named
   result section.
4. Output ONLY the Markdown content (no preamble, no postamble).
"""


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF, with page delimiters."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append(f"--- Page {page_num} ---\n{text}")
    doc.close()
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def generate_supplementary_plan(
    pdf_path: str,
    base_instruction: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_tokens: int = 16384,
    temperature: float = 1,
    tree_json_path: Optional[str] = None,
) -> str:
    """
    Read a paper PDF plus its existing base instruction and return only the
    supplementary detail (Markdown) that the base instruction does not cover.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY must be set in the environment or passed directly"
        )

    client = openai.OpenAI(api_key=api_key)
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text.strip():
        raise ValueError(f"No text could be extracted from {pdf_path}")

    user_message = (
        "BASE INSTRUCTION (already given to the agent — do NOT repeat this):\n"
        "```\n"
        f"{base_instruction.strip()}\n"
        "```\n\n"
        "---\n\n"
        "FULL PAPER TEXT:\n\n"
        f"{paper_text}"
    )

    if tree_json_path:
        tree_path = Path(tree_json_path)
        if tree_path.exists():
            with open(tree_path, encoding="utf-8") as f:
                tree = json.load(f)
            user_message += (
                "\n\n---\n\n"
                "RESEARCH-PROBLEM TREE (use to ensure all experiments are covered):\n\n"
                f"```json\n{json.dumps(tree, indent=2)}\n```"
            )
        else:
            print(f"[warn] Tree JSON not found at {tree_json_path}, skipping.")

    user_message += "\n\nNow write the supplementary experiment plan."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUPPLEMENTARY_PLAN_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )

    supplement = response.choices[0].message.content

    usage = response.usage
    print(
        f"Tokens  --  input: {usage.prompt_tokens}  |  "
        f"output: {usage.completion_tokens}  |  "
        f"total: {usage.total_tokens}"
    )

    return supplement


def build_instruction_gt(base_instruction: str, supplement: str) -> str:
    """Combine the base instruction with the supplementary plan."""
    return (
        base_instruction.rstrip()
        + "\n\n"
        + "---\n\n"
        + supplement.strip()
        + "\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate instruction_gt.txt by enriching an existing instruction.txt "
            "with detailed experimental plans extracted from the paper PDF."
        )
    )
    parser.add_argument(
        "paper_dir",
        help=(
            "Path to the benchmark paper folder (e.g. benchmark/papers/lost_in_the_middle). "
            "Must contain instruction/instruction.txt and a PDF in the folder root."
        ),
    )
    parser.add_argument(
        "--pdf", default=None,
        help="Explicit path to the paper PDF (auto-detected from paper_dir if omitted).",
    )
    parser.add_argument(
        "--model", default="gpt-4o", help="OpenAI model name (default: gpt-4o)"
    )
    parser.add_argument(
        "--tree", default=None,
        help="Optional path to a pre-computed problem tree JSON (from tree_parser.py).",
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output", default=None,
        help="Output path for instruction_gt.txt (default: <paper_dir>/instruction/instruction_gt.txt).",
    )
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    if not paper_dir.is_dir():
        raise NotADirectoryError(f"paper_dir not found: {paper_dir}")

    # Locate instruction.txt
    instruction_path = paper_dir / "instruction" / "instruction.txt"
    if not instruction_path.exists():
        raise FileNotFoundError(f"instruction.txt not found at {instruction_path}")
    base_instruction = instruction_path.read_text(encoding="utf-8")

    # Locate PDF
    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        pdfs = list(paper_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(
                f"No PDF found in {paper_dir}. Use --pdf to specify the path."
            )
        if len(pdfs) > 1:
            print(f"[warn] Multiple PDFs found; using {pdfs[0]}. Use --pdf to override.")
        pdf_path = pdfs[0]

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Output path
    output_path = Path(args.output) if args.output else (
        paper_dir / "instruction" / "instruction_gt.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Paper dir:    {paper_dir}")
    print(f"PDF:          {pdf_path}")
    print(f"Base instr.:  {instruction_path}")
    print(f"Output:       {output_path}")
    print()

    supplement = generate_supplementary_plan(
        str(pdf_path),
        base_instruction=base_instruction,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tree_json_path=args.tree,
    )

    instruction_gt = build_instruction_gt(base_instruction, supplement)
    output_path.write_text(instruction_gt, encoding="utf-8")

    print(f"\ninstruction_gt.txt saved to {output_path}")


if __name__ == "__main__":
    main()
