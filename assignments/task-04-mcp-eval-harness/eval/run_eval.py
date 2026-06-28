"""
eval/run_eval.py — Golden-dataset evaluation harness
=====================================================
Runs the full LangGraph + MCP agent against every entry in golden_dataset.json,
scores answers with an LLM judge, writes eval/report.md, and exits with:
  0  — aggregate pass rate >= 80%
  1  — aggregate pass rate < 80% (CI gate failure)

Usage:
    python -m eval.run_eval
    echo %ERRORLEVEL%   # Windows: 0 or 1
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

_HERE = Path(__file__).parent
_DATASET = _HERE / "golden_dataset.json"
_REPORT = _HERE / "report.md"
_PASS_RATE_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert evaluator assessing an AI assistant's answer.

Question: {question}

Expected facts that should appear in the answer (some or all):
{expected_facts}

Agent's answer:
{answer}

Score the answer on three dimensions, each from 1 to 5:

1. FAITHFULNESS (1-5): Does the answer stay grounded in what the retrieved documents say?
2. RELEVANCE (1-5): Does the answer actually address the question asked?
3. FACT_COVERAGE (1-5): How many of the expected facts are present?
   1=none covered, 3=~50% covered, 5=all covered.

Respond ONLY with a JSON object (no markdown, no extra text):
{{"faithfulness": 4, "relevance": 5, "fact_coverage": 3, "explanation": "Brief reason."}}
"""


def _score_answer(question: str, expected_facts: list[str], answer: str) -> dict:
    from agent.llm import get_llm

    llm = get_llm(temperature=0.0)
    facts_str = "\n".join(f"- {f}" for f in expected_facts)
    prompt = _JUDGE_PROMPT.format(
        question=question, expected_facts=facts_str, answer=answer
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines() if not line.strip().startswith("```")
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "faithfulness": 1,
            "relevance": 1,
            "fact_coverage": 1,
            "explanation": f"Judge returned non-JSON: {raw[:200]}",
        }


def _passes(scores: dict) -> bool:
    return (
        scores.get("faithfulness", 0) >= 3
        and scores.get("relevance", 0) >= 3
        and scores.get("fact_coverage", 0) >= 3
    )


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


async def run_eval() -> int:
    from agent.graph import agent_session

    dataset = json.loads(_DATASET.read_text(encoding="utf-8"))

    print(f"\n{'='*60}")
    print(f"Task-04 Evaluation Harness — {len(dataset)} questions")
    print(f"Pass threshold: {_PASS_RATE_THRESHOLD:.0%}")
    print(f"{'='*60}\n")

    print("Building agent (loading MCP tools)...")
    results = []

    async with agent_session() as run:
        print("Agent ready.\n")

        for idx, entry in enumerate(dataset, 1):
            qid = entry["id"]
            question = entry["question"]
            expected_facts = entry["expected_facts"]

            print(f"[{idx:02d}/{len(dataset)}] {qid}: {question[:70]}...")

            try:
                answer = await run(
                    question,
                    metadata={"mode": "eval", "question_id": qid},
                )
            except Exception as exc:
                answer = f"[AGENT ERROR: {exc}]"
                print(f"  x Agent error: {exc}")

            scores = _score_answer(question, expected_facts, answer)
            passed = _passes(scores)
            status = "PASS" if passed else "FAIL"
            print(
                f"  {status} | faith={scores['faithfulness']} "
                f"rel={scores['relevance']} "
                f"cov={scores['fact_coverage']}"
            )

            results.append(
                {
                    "id": qid,
                    "question": question,
                    "expected_facts": expected_facts,
                    "source_doc": entry.get("source_doc", ""),
                    "answer": answer,
                    "scores": scores,
                    "passed": passed,
                }
            )

    # Aggregate
    total = len(results)
    num_passed = sum(1 for r in results if r["passed"])
    pass_rate = num_passed / total if total else 0.0

    print(f"\n{'='*60}")
    print(f"Results: {num_passed}/{total} passed  ({pass_rate:.1%})")
    print(f"Threshold: {_PASS_RATE_THRESHOLD:.0%}")
    print("OUTCOME:", "PASS" if pass_rate >= _PASS_RATE_THRESHOLD else "FAIL")
    print(f"{'='*60}\n")

    _write_report(results, pass_rate)
    print(f"Report written to {_REPORT}\n")

    return 0 if pass_rate >= _PASS_RATE_THRESHOLD else 1


def _write_report(results: list[dict], pass_rate: float) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    num_passed = sum(1 for r in results if r["passed"])

    lines = [
        "# Task-04 Evaluation Report",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Questions evaluated:** {total}  ",
        f"**Passed:** {num_passed} / {total}  ",
        f"**Pass rate:** {pass_rate:.1%}  ",
        f"**Threshold:** {_PASS_RATE_THRESHOLD:.0%}  ",
        f"**Outcome:** {'PASS' if pass_rate >= _PASS_RATE_THRESHOLD else 'FAIL'}",
        "",
        "---",
        "",
        "## Per-Question Breakdown",
        "",
    ]

    for r in results:
        s = r["scores"]
        status = "PASS" if r["passed"] else "FAIL"
        lines += [
            f"### {r['id']} — {status}",
            "",
            f"**Question:** {r['question']}  ",
            f"**Source doc:** `{r['source_doc']}`  ",
            "",
            "**Expected facts:**",
            "",
            *[f"- {f}" for f in r["expected_facts"]],
            "",
            "**Scores:**",
            "",
            "| Dimension | Score (1-5) |",
            "|---|---|",
            f"| Faithfulness | {s.get('faithfulness', '?')} |",
            f"| Relevance | {s.get('relevance', '?')} |",
            f"| Fact coverage | {s.get('fact_coverage', '?')} |",
            "",
            f"**Judge explanation:** {s.get('explanation', '')}",
            "",
            "**Agent answer:**",
            "",
            textwrap.fill(r["answer"], width=100),
            "",
            "---",
            "",
        ]

    _REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    exit_code = asyncio.run(run_eval())
    sys.exit(exit_code)