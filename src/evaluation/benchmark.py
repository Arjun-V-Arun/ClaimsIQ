"""Benchmark ClaimsIQ tool routing, multi-tool orchestration and grounded facts."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from src.agent.groq_agent import ClaimsAgent
from src.rag.hybrid import normalise_token

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CaseResult:
    case_id: str
    expected_tools: list[str]
    actual_tools: list[str]
    tool_exact_match: bool
    tool_set_match: bool
    tool_recall: float
    answer_fact_match: bool
    latency_ms: float
    source_count: int
    tool_time_ms: float = 0.0
    llm_time_ms: float = 0.0
    llm_calls: int = 0


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise(text: str) -> str:
    text = text.lower().replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()


def _number_variants(value: str) -> set[str]:
    digits = re.sub(r"[^0-9.]", "", value)
    if not digits:
        return set()
    variants = {value.lower(), digits}
    try:
        if "." in digits:
            number = float(digits)
            variants.add(f"{number:g}")
        else:
            variants.add(f"{int(digits):,}")
    except ValueError:
        pass
    return variants


def _contains_variant(text: str, term: str) -> bool:
    norm = _normalise(term)
    if re.fullmatch(r"[\d,.$%.-]+", norm):
        return any(v in text for v in _number_variants(norm))

    # Match ordinary morphological variants such as claim/claims and
    # policy/policies without making phrase checks overly permissive.
    term_tokens = [normalise_token(t) for t in re.findall(r"[a-z0-9]+", norm)]
    text_tokens = [normalise_token(t) for t in re.findall(r"[a-z0-9]+", text)]
    if not term_tokens:
        return False
    if len(term_tokens) == 1:
        return term_tokens[0] in set(text_tokens)

    width = len(term_tokens)
    return any(text_tokens[i : i + width] == term_tokens for i in range(len(text_tokens) - width + 1))


def _group_match(answer_norm: str, group: list[str]) -> bool:
    return any(_contains_variant(answer_norm, term) for term in group)


def check_answer_facts(answer: str, case: dict) -> bool:
    """Deterministic evidence check with numeric and paraphrase variants."""
    answer_norm = _normalise(answer)
    required = [_normalise(term) for term in case.get("must_contain", [])]
    synonym_groups = case.get("must_contain_any", [])
    forbidden = [_normalise(term) for term in case.get("must_not_contain", [])]
    required_any_values = case.get("must_contain_values", [])

    return (
        all(_contains_variant(answer_norm, term) for term in required)
        and all(_group_match(answer_norm, group) for group in synonym_groups)
        and all(_group_match(answer_norm, group) for group in required_any_values)
        and not any(_contains_variant(answer_norm, term) for term in forbidden)
    )


def run_benchmark(agent: ClaimsAgent, cases: list[dict]) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        response = agent.answer(case["question"])
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        actual = [step["tool"] for step in response.get("tool_trace", [])]
        expected = case["expected_tools"]
        overlap = len(set(expected) & set(actual))
        recall = overlap / len(set(expected)) if expected else 1.0
        tool_time_ms = sum(float(step.get("latency_ms") or 0) for step in response.get("tool_trace", []))
        llm_time_ms = float(response.get("llm_total_ms") or 0.0)
        llm_calls = int(response.get("llm_calls") or 0)

        answer = response.get("answer", "")
        answer_fact_match = check_answer_facts(answer, case)

        results.append(
            CaseResult(
                case_id=case["id"],
                expected_tools=expected,
                actual_tools=actual,
                tool_exact_match=actual == expected,
                tool_set_match=set(actual) == set(expected),
                tool_recall=recall,
                answer_fact_match=answer_fact_match,
                latency_ms=latency_ms,
                source_count=len(response.get("sources", [])),
                tool_time_ms=round(tool_time_ms, 2),
                llm_time_ms=round(llm_time_ms, 2),
                llm_calls=llm_calls,
            )
        )
    return results


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    frac = position - lower
    return values[lower] + (values[upper] - values[lower]) * frac


def summarise(results: list[CaseResult]) -> dict[str, float]:
    if not results:
        return {}
    multi = [r for r in results if len(r.expected_tools) > 1]
    latencies = [r.latency_ms for r in results]
    return {
        "tool_exact_match_rate": sum(r.tool_exact_match for r in results) / len(results),
        "tool_set_match_rate": sum(r.tool_set_match for r in results) / len(results),
        "tool_recall": sum(r.tool_recall for r in results) / len(results),
        "answer_fact_match_rate": sum(r.answer_fact_match for r in results) / len(results),
        "mean_latency_ms": sum(latencies) / len(results),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "mean_tool_time_ms": sum(r.tool_time_ms for r in results) / len(results),
        "mean_llm_trace_time_ms": sum(r.llm_time_ms for r in results) / len(results),
        "mean_llm_calls": sum(r.llm_calls for r in results) / len(results),
        "source_presence_rate": sum(r.source_count > 0 for r in results) / len(results),
        "multi_tool_cases": len(multi),
        "multi_tool_set_match_rate": (sum(r.tool_set_match for r in multi) / len(multi)) if multi else 1.0,
    }


def main() -> None:
    import argparse
    import os
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "claims_real.db")
    parser.add_argument("--policy-dir", type=Path, default=ROOT / "data" / "chroma")
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "benchmark.json")
    args = parser.parse_args()

    from src.agent.tools import ClaimsTools

    tools = ClaimsTools(args.db, args.policy_dir)
    warmup_started = time.perf_counter()
    if any("search_policy" in case["expected_tools"] for case in load_cases(args.cases)):
        tools.warmup_rag()
    rag_warmup_ms = round((time.perf_counter() - warmup_started) * 1000, 2)

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")

    agent = ClaimsAgent(tools)
    results = run_benchmark(agent, load_cases(args.cases))
    summary = summarise(results)
    summary["rag_warmup_ms"] = rag_warmup_ms

    payload = {"summary": summary, "cases": [r.__dict__ for r in results]}
    failed = [r.case_id for r in results if not r.answer_fact_match]
    if failed:
        payload["answer_fact_failures"] = failed
        payload["note"] = "Inspect these cases manually; the fact checker is deterministic and does not judge semantic correctness."
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
