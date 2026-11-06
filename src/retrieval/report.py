"""Generate docs/retrieval.md and the README table from eval/retrieval/.

Published numbers are rendered from result files, never typed, and CI checks
that the two agree.
"""
from __future__ import annotations

import json

from src.retrieval import data
from src.retrieval.experiments import RESULTS_DIR

DOC = data.ROOT / "docs" / "retrieval.md"
README = data.ROOT / "README.md"
START, END = "<!-- retrieval:start -->", "<!-- retrieval:end -->"
DENSE_PENDING = ("*Not run yet. Embedding, fusion, and reranking results come from "
                 "`python -m src.retrieval eval --dense` on a machine that can download "
                 "models.*")

NAMES = {
    "bm25": "BM25 keyword search",
    "dense_current": "Embeddings, the copilot's current model",
    "dense_best": "Embeddings, best model on valid",
    "hybrid": "Hybrid: BM25 + embeddings, rank fusion",
    "hybrid_rerank": "Hybrid + cross-encoder rerank",
}


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(value: float | None, digits: int = 1) -> str:
    return "" if value is None else f"{100 * value:.{digits}f}%"


def _ci(m: dict, pct: bool = True) -> str:
    if pct:
        return f"{_pct(m['estimate'])} ({_pct(m['low'])} to {_pct(m['high'])})"
    return f"{m['estimate']:.3f} ({m['low']:.3f} to {m['high']:.3f})"


def _delta(m: dict, pct: bool = True) -> str:
    scale, unit = (100, " pts") if pct else (1, "")
    fmt = "{:+.1f}" if pct else "{:+.3f}"
    return (f"{fmt.format(scale * m['estimate'])}{unit} "
            f"({fmt.format(scale * m['low'])} to {fmt.format(scale * m['high'])})")


def summary_table() -> str:
    dataset, bm25, systems, gate = (_load("dataset"), _load("bm25"),
                                    _load("systems"), _load("abstention"))
    rows = []
    if dataset and bm25:
        rows.append(("Benchmark", f"{dataset['answers']:,} expert answers, "
                     f"{bm25['test_questions']:,} real test questions",
                     "InsuranceQA V2, labels from the dataset authors"))
        m = bm25["metrics"]
        rows.append(("BM25: correct answer in top 5", _ci(m["success@5"]),
                     f"MRR@10 {m['mrr@10']['estimate']:.3f}, "
                     f"{bm25['latency']['p50_ms']:.1f} ms per query on CPU"))
    if systems:
        best = systems["systems"]["hybrid_rerank"]
        delta = systems["paired"]["hybrid_rerank vs bm25"]["success@5"]
        rows.append(("Hybrid + rerank: correct answer in top 5", _ci(best["metrics"]["success@5"]),
                     f"{_delta(delta)} vs BM25, "
                     f"{best['latency']['p50_ms']:.0f} ms per query on {systems['device']}"))
    if gate:
        old = gate["original_gate"]
        new = gate["tuned_rerank_gate"]
        rows.append(("Original relevance gate on real unanswerable questions",
                     f"refused {_pct(old['refused_when_unanswerable'], 0)}",
                     f"and refused {_pct(old['refused_when_answerable'], 0)} of answerable ones"))
        rows.append(("Tuned gate on the same questions",
                     f"refused {_pct(new['refused_when_unanswerable'], 0)}",
                     f"and refused {_pct(new['refused_when_answerable'], 0)} of answerable ones"))
    if not rows:
        return "*No results yet. Run `python -m src.retrieval build` and `eval`.*"
    lines = ["| Measure | Result | Detail |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    if not systems:
        lines += ["", DENSE_PENDING]
    return "\n".join(lines)


def _gate_rows(name: str, g: dict) -> str:
    return (f"| {name} | {_pct(g['refused_when_unanswerable'])} "
            f"| {_pct(g['refused_when_answerable'])} | {_pct(g['success@5_all_answerable'])} "
            f"| {_pct(g['success@5_accepted_answerable'])} |")


def document() -> str:
    out = ["# Retrieval Benchmark", "",
           "Generated from `eval/retrieval/` by `python -m src.retrieval report`. "
           "Do not edit by hand.", "",
           "Every setting below was chosen on the valid split by a rule fixed in advance "
           "(highest MRR@10 unless stated). The test split was used only to report.", ""]

    dataset = _load("dataset")
    if dataset:
        q = dataset["questions"]
        out += ["## Data", "",
                f"- {dataset['answers']:,} answers written by insurance professionals, "
                f"median {dataset['answer_median_words']} words.",
                f"- {sum(q.values()):,} questions from real users: {q['train']:,} train, "
                f"{q['valid']:,} valid, {q['test']:,} test, as published.",
                f"- {dataset['questions_with_several_answers']:,} questions have more than one "
                "correct answer, so two families of metric are kept apart. Success@k: is any "
                "correct answer in the top k? Recall@k: what share of them are?",
                f"- {len(dataset['domains'])} domains. Source: {dataset['source']}. Released "
                "for research use only, so it is downloaded at build time and not committed. "
                "Feng et al., ASRU 2015.", ""]

    selection = _load("selection_bm25")
    bm25 = _load("bm25")
    if bm25 and selection:
        p = selection["chosen"]
        m = bm25["metrics"]
        pools = bm25["keyword_pools"]
        out += ["## Keyword Baseline", "",
                f"BM25 with k1 = {p['k1']}, b = {p['b']}, stemming "
                f"{'on' if p['stem'] else 'off'}, chosen from {len(selection['grid'])} settings.", "",
                "| Metric | Test (95% CI) |", "|---|---|"]
        for name in ("success@1", "success@5", "success@20", "success@100",
                     "recall@20", "recall@100"):
            out.append(f"| {name.replace('success', 'Success').replace('recall', 'Recall')} "
                       f"| {_ci(m[name])} |")
        for name in ("mrr@10", "ndcg@10"):
            out.append(f"| {name.upper().replace('@', '@')} | {_ci(m[name], pct=False)} |")
        lat = bm25["latency"]
        out += ["", f"Latency: {lat['p50_ms']} ms median, {lat['p95_ms']} ms p95, one query "
                f"at a time on {lat['device']}.", "",
                "The dataset ships a 100-answer candidate pool per question from keyword "
                "search. It contains a correct answer for "
                f"{_pct(pools['solr_pool_has_a_correct_answer'])} of test questions and every "
                f"correct answer for {_pct(pools['solr_pool_has_every_correct_answer'])}. This "
                "BM25's top 100 reaches "
                f"{_pct(pools['our_bm25_top100_has_a_correct_answer'])} and "
                f"{_pct(pools['our_bm25_top100_has_every_correct_answer'])}. Keyword search "
                "alone misses many correct answers entirely, which is the gap learned "
                "retrieval has to close.", ""]

    out += ["## Learned Retrieval", ""]
    systems = _load("systems")
    chosen = _load("selection_dense")
    if systems and chosen:
        out += [f"Run on {systems['device']}. Embedding model chosen on valid: "
                f"`{chosen['chosen_model']}` ("
                + ", ".join(f"`{k}` {v:.3f}" for k, v in chosen["dense_models"].items())
                + f" MRR@10). Fusion constant k = {chosen['chosen_rrf_k']}. Reranker "
                f"`{chosen['reranker']}` over the top {chosen['chosen_rerank_depth']}.", "",
                "| System | Success@1 | Success@5 (95% CI) | MRR@10 | nDCG@10 | Success@100 "
                "| Latency p50 / p95 |", "|---|---|---|---|---|---|---|"]
        for key, sys_ in systems["systems"].items():
            m = sys_["metrics"]
            lat = sys_["latency"]
            out.append(f"| {NAMES.get(key, key)} | {_pct(m['success@1']['estimate'])} "
                       f"| {_ci(m['success@5'])} | {m['mrr@10']['estimate']:.3f} "
                       f"| {m['ndcg@10']['estimate']:.3f} | {_pct(m['success@100']['estimate'])} "
                       f"| {lat['p50_ms']:.0f} / {lat['p95_ms']:.0f} ms |")
        out += ["", "Paired differences on the same test questions, 95% intervals:", "",
                "| Comparison | Success@5 | MRR@10 | Questions better / worse |",
                "|---|---|---|---|"]
        for name, d in systems["paired"].items():
            a, b = name.split(" vs ")
            out.append(f"| {NAMES.get(a, a)} vs {NAMES.get(b, b)} | {_delta(d['success@5'])} "
                       f"| {_delta(d['mrr@10'], pct=False)} "
                       f"| {d['success@5']['better']} / {d['success@5']['worse']} |")
        dom = systems["by_domain"]
        out += ["", "Success@5 by domain:", "", "| Domain | Questions | BM25 | Hybrid + rerank |",
                "|---|---|---|---|"]
        for domain, row in dom["bm25"].items():
            out.append(f"| {domain} | {row['n']} | {_pct(row['success@5'])} "
                       f"| {_pct(dom['hybrid_rerank'][domain]['success@5'])} |")
        out += ["", "Questions hybrid + rerank missed entirely (no correct answer in its top "
                "20), a seeded sample:", ""]
        for f in systems["failures"]:
            out += [f"- **{f['question']}** ({f['domain']})",
                    f"  - Retrieved first: {f['retrieved_first']}",
                    f"  - A correct answer: {f['a_correct_answer']}"]
        out.append("")
    else:
        out += [DENSE_PENDING, ""]

    out += ["## Abstention", ""]
    gate_bm25 = _load("abstention_bm25")
    gate = _load("abstention")
    if gate_bm25:
        held = gate_bm25["held_out"]
        qs = gate_bm25["questions"]["test"]
        out += ["A retriever should refuse when the corpus has no answer. To test that with "
                f"real questions, every answer belonging to **{held['domain']}** is removed "
                f"from the index ({gate_bm25['answers_removed']:,} answers), making its "
                f"{qs['unanswerable']} test questions unanswerable, while {qs['answerable']:,} "
                f"questions from other domains stay answerable. The domain was chosen by rule, "
                f"on the train split only: {held['rule']}. Thresholds are tuned on valid for "
                "balanced accuracy.", "",
                "| Gate | Refused, unanswerable | Refused, answerable | Success@5, all "
                "answerable | Success@5, accepted |", "|---|---|---|---|---|"]
        if gate:
            out.append(_gate_rows("Copilot's original rule and thresholds", gate["original_gate"]))
            out.append(_gate_rows("Tuned on top cosine, current model", gate["tuned_cosine_gate"]))
            out.append(_gate_rows("Tuned on top cross-encoder score", gate["tuned_rerank_gate"]))
        out.append(_gate_rows("Tuned on BM25 score per unit of query idf", gate_bm25["bm25"]))
        out += ["", "Removing a domain leaves related answers in other domains, so some "
                "\"unanswerable\" questions still have partly useful answers. Refusal rates "
                "on them are a lower bound on what an ideal gate would do.", ""]

    out += ["## Limits", "",
            "- One public dataset from one Q&A site, mostly about US insurance products.",
            "- Labels mark the answers written for each question. Another answer can be "
            "correct without being labelled, so every score here is a lower bound.",
            "- No model was fine-tuned on the training split. Doing so would likely raise "
            "every learned score.",
            "- Latency is measured one query at a time on one machine and will differ "
            "elsewhere.",
            "- The copilot's `search_policy` tool still uses the four demonstration "
            "documents. Switching it to the winning retriever is the next step.", ""]
    return "\n".join(out)


def _readme() -> str:
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise RuntimeError("README is missing the retrieval markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{summary_table()}\n{END}{tail}"


def write() -> None:
    DOC.write_text(document() + "\n", encoding="utf-8")
    README.write_text(_readme(), encoding="utf-8")
    print("wrote docs/retrieval.md and the README table")


def check() -> int:
    stale = []
    if not DOC.exists() or DOC.read_text(encoding="utf-8") != document() + "\n":
        stale.append("docs/retrieval.md")
    if README.read_text(encoding="utf-8") != _readme():
        stale.append("README.md")
    if stale:
        print(f"Out of date with eval/retrieval: {', '.join(stale)}. "
              "Run `python -m src.retrieval report`.")
        return 1
    print("Published retrieval numbers match the result files.")
    return 0
