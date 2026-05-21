import json
import re
import argparse
import os
from pathlib import Path
from typing import Optional

import fitz
import openai
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TREE_PARSING_PROMPT = """\
You are a research-paper expert specializing in methodological analysis and problem
decomposition of scientific studies.

**GOAL**

- Fully comprehend the paper, understand its core research problems and experiments
- Then, parse the given paper and construct a hierarchical **research-problem tree**
  that mirrors the authors' logic as follows:

  * **Root node** -- the single, more essential, broadest research problem tackled by
    the paper.
  * **Intermediate nodes** -- progressively narrower sub-problems/questions/objectives
    that the authors introduce to tackle the root.
  * **Leaf nodes** -- fully specified experimental tasks (datasets, models, metrics, or
    protocols) that map to a *figure, table, or named result section* in the paper.
    Continue decomposing until every branch ends in such a leaf. There is no depth limit.

---

### Reading & Extraction Rules

1. **Locate the root** in the title, abstract, introduction, or discussion.
2. **Recursively decompose** each problem by following explicit textual cues
   (headings, "first... second...", "to this end...", method overviews, figure/table
   captions, bullet lists, etc.).
3. **Identify leaves**: a node is a leaf *only if* it describes a concrete experiment
   and you can cite the corresponding Figure / Table / Section ID.
4. **Capture all layers**--do **not** skip intermediate hypotheses, objectives, or
   analysis steps the paper explicitly discusses.
5. **Stay faithful** to the paper's wording for technical terms; paraphrase only for
   brevity or clarity.
6. **No outside invention**--derive every node from the paper alone. If information
   is missing, mark the node with [uncertain].

Strictly output the tree in a JSON format:

```json
{
  "paper": {
    "title": "",
    "authors": [],
    "venue": "",
    "year": ""
  },
  "problem_tree": {
    "node": "Root: broadest research problem tackled by the paper",
    "type": "root node",
    "description": "a detailed description of the research problem in this node",
    "evidence": "references back to the original paper to back up the construction of this node",
    "children": [
      {
        "node": "Intermediate sub-problem / objective 1",
        "type": "depth-1 node",
        "description": "a detailed description of the research problem in this node",
        "evidence": "references back to the original paper to back up the construction of this node",
        "children": [
          {
            "node": "Narrower question or method component",
            "type": "depth-2 node",
            "description": "a detailed description of the research problem in this node",
            "evidence": "references back to the original paper to back up the construction of this node",
            "children": [
              {
                "type": "leaf node",
                "task": "Concrete experimental task (as phrased by paper)",
                "dataset": ["..."],
                "model_or_method": ["..."],
                "metrics": ["..."],
                "protocol_or_setup": "key settings/splits/hyperparams if stated",
                "evidence": {
                  "figure": "Fig. X",
                  "table": "Table Y",
                  "section": "Sec. Z or Result subsection name"
                },
                "conclusion": "explicit and detailed conclusions derived from experiments in this current leaf node",
                "status": ""  // leave empty or set to "[uncertain]" if any item is missing in the paper
              }
            ]
          }
        ]
      },
      {
        "node": "Intermediate sub-problem / objective 2",
        "children": [ /* ...more branches ending in leaves... */ ]
      }
    ]
  }
}
```
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
# Response parsing
# ---------------------------------------------------------------------------

def extract_json_from_response(text: str) -> dict:
    """Pull the first JSON object out of a model response, handling markdown fences."""
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start : end + 1])
    raise ValueError("No valid JSON found in the model response")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def parse_paper_to_tree(
    pdf_path: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_tokens: int = 16384,
    temperature: float = 0.0,
) -> dict:
    """Read a paper PDF and return the research-problem tree as a dict."""
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
        "Below is the full text of the paper:\n\n"
        f"{paper_text}\n\n"
        "Now parse this paper according to the instructions."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TREE_PARSING_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    raw_output = response.choices[0].message.content
    tree = extract_json_from_response(raw_output)

    usage = response.usage
    print(
        f"Tokens  --  input: {usage.prompt_tokens}  |  "
        f"output: {usage.completion_tokens}  |  "
        f"total: {usage.total_tokens}"
    )

    return tree


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse a research paper (PDF) into a problem tree via OpenAI."
    )
    parser.add_argument("pdf_path", help="Path to the paper PDF")
    parser.add_argument(
        "--model", default="gpt-4o", help="OpenAI model name (default: gpt-4o)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output JSON path (default: <pdf_stem>_tree.json)",
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_path = args.output or str(pdf_path.with_suffix("")) + "_tree.json"

    tree = parse_paper_to_tree(
        str(pdf_path),
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)

    print(f"Tree saved to {output_path}")


if __name__ == "__main__":
    main()
