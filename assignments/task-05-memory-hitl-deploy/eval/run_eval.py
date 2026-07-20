"""
eval/run_eval.py — Extended evaluation harness
===============================================
Handles three question types:
  standard      — same as Task 04 single-turn RAG eval
  memory_set    — turn 1: set a preference; check agent acknowledged it
  memory_recall — turn 2: ask a question; check preference was applied
  memory_episode — check episodic memory recall

Exit codes:
  0 — pass rate >= 80%
  1 — pass rate < 80%

Usage:
    python -m eval.run_eval
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
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

_HERE = Path(__file__).parent
_DATASET = _HERE / "golden_dataset.json"
_REPORT = _HERE / "report.md"
_PASS_RATE_THRESHOLD = 0.80

_JUDGE_PROMPT = """\
You are an expert evaluator assessing an AI assistant's answer.

Question: {question}
Question type: {qtype}
Expected facts/elements: {expected_facts}
Agent's answer: {answer}
{extra_context}

Score on three dimensions (1-5 each):
1. FAITHFULNESS: Is the answer grounded and accurate?
2. RELEVANCE: Does it address the question?
3. FACT_COVERAGE: How many expected facts/elements appear?
   1=none, 3=~50%, 5=all present.

{special_instructions}

Respond ONLY with JSON (no markdown):
{{"faithfulness": 4, "relevance": 5, "fact_coverage": 3, "explanation": "reason"}}
"""


def _score_answer(
    question: str,
    expected_facts: list[str],
    answer: str,
    qtype: str = "standard",
    extra_context: str = "",
    special_instructions: str = "",
) -> dict:
    from agent.llm import get_llm

    llm = get_llm(temperature=0.0)
    facts_str = "\n".join(f"- {f}" for f in expected_facts)
    prompt = _JUDGE_PROMPT.format(
        question=question,
        qtype=qtype,
        expected_facts=facts_str,
        answer=answer,
        extra_context=extra_context,
        special_instructions=special_instructions,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```"))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"faithfulness": 1, "relevance": 1, "fact_coverage": 1,
                "explanation": f"Non-JSON: {raw[:200]}"}


def _passes(scores: dict) -> bool:
    return (
        scores.get("faithfulness", 0) >= 3
        and scores.get("relevance", 0) >= 3
        and scores.get("fact_coverage", 0) >= 3
    )


async def run_eval() -> int:
    from langgraph.store.memory import InMemoryStore
    from agent.graph import _get_builder, CONFIRMABLE_TOOLS

    dataset = json.loads(_DATASET.read_text(encoding="utf-8"))

    print(f"\n{'='*60}")
    print(f"Task-05 Evaluation Harness — {len(dataset)} questions")
    print(f"Pass threshold: {_PASS_RATE_THRESHOLD:.0%}")
    print(f"{'='*60}\n")

    store = InMemoryStore()

    client = MultiServerMCPClient(
        {
            "task04_tools": {
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
                "env": {"PYTHONPATH": str(Path(__file__).parent.parent)},
            },
            "time": {
                "command": sys.executable,
                "args": ["-m", "mcp_server_time", "--local-timezone=UTC"],
                "transport": "stdio",
            },
        }
    )

    all_tools = []
    sessions = []

    print("Loading MCP tools...")
    try:
        for server_name in ["task04_tools", "time"]:
            cm = client.session(server_name)
            session = await cm.__aenter__()
            sessions.append((cm, session))
            tools = await load_mcp_tools(session)
            all_tools.extend(tools)

        builder = _get_builder(all_tools, store)
        graph = builder.compile()
        print("Agent ready.\n")

        results = []
        # Track memory between m01→m02 and m03→m04 pairs
        memory_context: dict = {}

        for idx, entry in enumerate(dataset, 1):
            qid = entry["id"]
            question = entry["question"]
            expected_facts = entry["expected_facts"]
            qtype = entry.get("type", "standard")

            print(f"[{idx:02d}/{len(dataset)}] {qid} ({qtype}): {question[:60]}...")

            user_id = f"eval_{qtype}"
            import uuid
            thread_id = str(uuid.uuid4())

            from langchain_core.runnables import RunnableConfig
            config = RunnableConfig(
                metadata={"mode": "eval" if qtype == "standard" else "eval_memory",
                          "question_id": qid},
                run_name=f"task05-eval-{qid}",
                configurable={"thread_id": thread_id},
            )

            initial_state = {
                "messages": [HumanMessage(content=question)],
                "user_id": user_id,
                "memory_context": memory_context.get(qid, ""),
                "pending_interrupt": None,
                "run_metadata": {"mode": "eval", "question_id": qid},
                "tool_results": [],
            }

            try:
                result = await graph.ainvoke(initial_state, config=config)
                answer = result["messages"][-1].content
                # Store answer for dependent questions
                memory_context[qid] = answer
            except Exception as exc:
                answer = f"[AGENT ERROR: {exc}]"
                print(f"  x Error: {exc}")

            # Build special judge instructions per type
            special = ""
            extra = ""
            if qtype == "memory_recall":
                check_pref = entry.get("check_preference", "")
                special = (
                    f"IMPORTANT: Also check whether the answer applies the user preference "
                    f"'{check_pref}'. If the preference is not reflected, lower fact_coverage."
                )
                depends = entry.get("depends_on", "")
                if depends in memory_context:
                    extra = f"Note: User previously stated preference in turn '{depends}': {memory_context[depends][:200]}"
            elif qtype == "memory_episode":
                special = (
                    "Check if the answer references or builds on past conversation context. "
                    "Award fact_coverage=5 if the agent demonstrates awareness of session history."
                )

            scores = _score_answer(
                question, expected_facts, answer, qtype, extra, special
            )
            passed = _passes(scores)
            status = "PASS" if passed else "FAIL"
            print(f"  {status} | faith={scores['faithfulness']} rel={scores['relevance']} cov={scores['fact_coverage']}")

            results.append({
                "id": qid, "question": question,
                "expected_facts": expected_facts,
                "source_doc": entry.get("source_doc", ""),
                "type": qtype, "answer": answer,
                "scores": scores, "passed": passed,
            })

    finally:
        for cm, session in reversed(sessions):
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass

    total = len(results)
    num_passed = sum(1 for r in results if r["passed"])
    pass_rate = num_passed / total if total else 0.0

    print(f"\n{'='*60}")
    print(f"Results: {num_passed}/{total} passed ({pass_rate:.1%})")
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
        "# Task-05 Evaluation Report", "",
        f"**Generated:** {timestamp}  ",
        f"**Questions evaluated:** {total}  ",
        f"**Passed:** {num_passed} / {total}  ",
        f"**Pass rate:** {pass_rate:.1%}  ",
        f"**Threshold:** {_PASS_RATE_THRESHOLD:.0%}  ",
        f"**Outcome:** {'PASS' if pass_rate >= _PASS_RATE_THRESHOLD else 'FAIL'}",
        "", "---", "", "## Per-Question Breakdown", "",
    ]

    for r in results:
        s = r["scores"]
        status = "PASS" if r["passed"] else "FAIL"
        lines += [
            f"### {r['id']} ({r['type']}) — {status}", "",
            f"**Question:** {r['question']}  ",
            f"**Source doc:** `{r['source_doc']}`  ",
            "", "**Expected facts:**", "",
            *[f"- {f}" for f in r["expected_facts"]],
            "", "**Scores:**", "",
            "| Dimension | Score (1-5) |", "|---|---|",
            f"| Faithfulness | {s.get('faithfulness', '?')} |",
            f"| Relevance | {s.get('relevance', '?')} |",
            f"| Fact coverage | {s.get('fact_coverage', '?')} |",
            "", f"**Judge explanation:** {s.get('explanation', '')}",
            "", "**Agent answer:**", "",
            textwrap.fill(r["answer"], width=100),
            "", "---", "",
        ]

    _REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    exit_code = asyncio.run(run_eval())
    sys.exit(exit_code)
