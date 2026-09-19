from src.evaluation.benchmark import CaseResult, check_answer_facts, summarise


def test_benchmark_summary():
    results = [
        CaseResult("a", ["x"], ["x"], True, True, 1.0, True, 100.0, 1),
        CaseResult("b", ["x"], ["y"], False, False, 0.0, False, 300.0, 0),
    ]
    summary = summarise(results)
    assert summary["tool_exact_match_rate"] == 0.5
    assert summary["tool_set_match_rate"] == 0.5
    assert summary["tool_recall"] == 0.5
    assert summary["answer_fact_match_rate"] == 0.5
    assert summary["mean_latency_ms"] == 200.0
    assert summary["source_presence_rate"] == 0.5
    assert summary["p95_latency_ms"] == 290.0


def test_answer_fact_checks():
    assert check_answer_facts(
        "R94 has the highest frequency. This is observed dataset data.",
        {"must_contain": ["R94", "observed dataset"], "must_not_contain": ["Assurant data"]},
    )
    assert not check_answer_facts(
        "R21 has the highest frequency.",
        {"must_contain": ["R94"]},
    )


def test_percentile_summary_interpolates():
    results = [
        CaseResult("a", ["x"], ["x"], True, True, 1.0, True, 100.0, 1),
        CaseResult("b", ["x"], ["x"], True, True, 1.0, True, 300.0, 1),
        CaseResult("c", ["x"], ["x"], True, True, 1.0, True, 500.0, 1),
    ]
    summary = summarise(results)
    assert summary["p95_latency_ms"] == 480.0
