"""Retrieval benchmark: data contract, retrievers, metrics, gates, and reports.

Uses a tiny dataset written in InsuranceQA's own file format, so the loader is
tested on the real encoding without downloading anything.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math

import numpy as np
import pytest

from src.rag.hybrid import accept_hybrid
from src.retrieval import bm25, data, dense, experiments, metrics, report

ANSWERS = {
    1: "Term life insurance covers you for a fixed term and pays a death benefit.",
    2: "Whole life insurance builds cash value over your lifetime.",
    3: "Medicare Part B covers doctor visits and outpatient care.",
    4: "Medicare Part D helps pay for prescription drugs.",
    5: "Renters insurance covers your belongings after a theft or fire.",
    6: "Auto insurance liability pays for damage you cause to other drivers.",
    7: "Collision coverage pays to repair your own car after an accident.",
    8: "Home insurance covers the structure of your house after storm damage.",
}
QUESTIONS = {
    "train": [("life-insurance", "What is term life insurance?", "1", "1 2 3"),
              ("medicare-insurance", "Does Medicare cover doctor visits?", "3", "3 4"),
              ("auto-insurance", "Who pays for damage I cause driving?", "6", "6 7")],
    "valid": [("life-insurance", "Does whole life build cash value?", "2", "1 2"),
              ("medicare-insurance", "Does Medicare pay for prescription drugs?", "4", "3 4"),
              ("renters-insurance", "Does renters insurance cover theft?", "5", "5 8")],
    "test": [("life-insurance", "How does term life insurance pay?", "1 2", "1 2 3"),
             ("medicare-insurance", "What does Medicare Part B cover?", "3", "4 3"),
             ("auto-insurance", "Will collision coverage repair my car?", "7", "6 7"),
             ("home-insurance", "Is storm damage to my house covered?", "8", "8 5")],
}


def _write_fixture(directory):
    words = sorted({w for t in [*ANSWERS.values(), *(q[1] for r in QUESTIONS.values() for q in r)]
                    for w in t.split()})
    index = {w: f"idx_{i}" for i, w in enumerate(words)}
    encode = lambda text: " ".join(index[w] for w in text.split())
    (directory / data.VOCABULARY).write_text(
        "".join(f"{v}\t{k}\n" for k, v in index.items()), encoding="utf-8")
    with gzip.open(directory / data.ANSWERS, "wt", encoding="utf-8") as out:
        for label, text in ANSWERS.items():
            out.write(f"{label}\t{encode(text)}\n")
    for split, rows in QUESTIONS.items():
        with gzip.open(directory / data.SPLIT_FILE.format(split=split), "wt",
                       encoding="utf-8") as out:
            for domain, text, gold, pool in rows:
                out.write(f"{domain}\t{encode(text)}\t{gold}\t{pool}\n")
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


@pytest.fixture
def fixture_dir(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    hashes = _write_fixture(source)
    return source, hashes


@pytest.fixture
def ds(fixture_dir):
    return data.load(fixture_dir[0], expected=None)


@pytest.fixture
def results(tmp_path, monkeypatch):
    out = tmp_path / "results"
    monkeypatch.setattr(experiments, "RESULTS_DIR", out)
    monkeypatch.setattr(report, "RESULTS_DIR", out)
    monkeypatch.setattr(experiments, "MIN_GATE_QUESTIONS", 1)
    return out


# --- Data contract -------------------------------------------------------------------

def test_loader_decodes_the_real_file_format(ds):
    assert ds.answers[3] == ANSWERS[3]
    question = ds.split("test")[0]
    assert question.text == "How does term life insurance pay?"
    assert question.answers == (1, 2) and question.pool == (1, 2, 3)
    assert question.qid == "test-0" and question.domain == "life-insurance"


def test_fetch_downloads_and_verifies(fixture_dir, tmp_path):
    source, hashes = fixture_dir
    target = tmp_path / "target"
    data.fetch(target, hashes, source.as_uri() + "/")
    assert sorted(p.name for p in target.iterdir()) == sorted(hashes)


def test_fetch_rejects_a_changed_file(fixture_dir, tmp_path):
    source, hashes = fixture_dir
    target = tmp_path / "target"
    data.fetch(target, hashes, source.as_uri() + "/")
    (target / data.VOCABULARY).write_text("tampered", encoding="utf-8")
    with pytest.raises(data.DataContractError):
        data.fetch(target, hashes, source.as_uri() + "/")


def test_unexpected_sizes_are_refused(fixture_dir):
    with pytest.raises(data.DataContractError, match="unexpected sizes"):
        data.load(fixture_dir[0], expected={"answers": 27_413})


def test_a_question_citing_an_unknown_answer_is_refused(fixture_dir):
    path = fixture_dir[0] / data.SPLIT_FILE.format(split="test")
    lines = gzip.open(path, "rt", encoding="utf-8").read().splitlines()
    fields = lines[0].split("\t")
    fields[2] = "999"
    lines[0] = "\t".join(fields)
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")
    with pytest.raises(data.DataContractError, match="unknown answers"):
        data.load(fixture_dir[0], expected=None)


def test_summary_counts_multi_answer_questions(ds):
    assert data.summary(ds)["questions_with_several_answers"] == 1


# --- BM25 ----------------------------------------------------------------------------

def _reference_bm25(query, docs, k1, b, analyze):
    tokenized = [analyze(d) for d in docs]
    avgdl = sum(map(len, tokenized)) / len(tokenized)
    n = len(docs)
    scores = []
    for words in tokenized:
        total = 0.0
        for term in set(analyze(query)):
            df = sum(term in d for d in tokenized)
            tf = words.count(term)
            if df == 0 or tf == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            total += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(words) / avgdl))
        scores.append(total)
    return scores


@pytest.mark.parametrize("stem", [False, True])
def test_bm25_matches_a_direct_implementation(ds, stem):
    ids = ds.answer_ids
    docs = [ds.answers[i] for i in ids]
    model = bm25.BM25(bm25.Params(1.2, 0.75, stem)).fit(ids, docs)
    query = "Does Medicare cover prescription drugs and doctor visits?"
    expected = _reference_bm25(query, docs, 1.2, 0.75, bm25.analyzer(stem))
    assert model.scores([query])[0] == pytest.approx(expected)


def test_reweighting_equals_refitting(ds):
    ids = ds.answer_ids
    docs = [ds.answers[i] for i in ids]
    model = bm25.BM25(bm25.Params(1.2, 0.75)).fit(ids, docs)
    model.reweight(0.9, 0.4)
    fresh = bm25.BM25(bm25.Params(0.9, 0.4)).fit(ids, docs)
    assert np.allclose(model.scores(["car accident repair"]), fresh.scores(["car accident repair"]))


def test_ties_go_to_the_lower_answer_id():
    model = bm25.BM25().fit([10, 4, 7], ["same text here", "same text here", "other words"])
    ranked, _ = model.search(["same text"], 3)
    assert list(ranked[0][:2]) == [4, 10]


def test_bm25_finds_the_obvious_answer(ds):
    model = experiments.fit_bm25(ds, bm25.Params())
    ranked, _ = model.search(["collision coverage repair my car"], 3)
    assert ranked[0][0] == 7


# --- Metrics -------------------------------------------------------------------------

def test_metrics_on_hand_computed_cases():
    ranked = np.array([[5, 1, 9, 2, 8, 0, 0, 0, 0, 0],
                       [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]])
    scores = metrics.per_question(ranked, [(1, 2), (3,)])
    assert list(scores["success@1"]) == [0, 0]
    assert list(scores["success@5"]) == [1, 0]
    assert scores["mrr@10"][0] == pytest.approx(0.5) and scores["mrr@10"][1] == 0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 + 1 / math.log2(3)
    assert scores["ndcg@10"][0] == pytest.approx(dcg / ideal)


def test_recall_counts_each_correct_answer():
    ranked = np.tile(np.arange(100), (1, 1))
    scores = metrics.per_question(ranked, [(3, 50, 500)])
    assert scores["recall@20"][0] == pytest.approx(1 / 3)
    assert scores["recall@100"][0] == pytest.approx(2 / 3)
    assert scores["success@20"][0] == 1


def test_intervals_and_paired_differences():
    values = {"success@5": np.array([1.0, 0.0] * 50)}
    interval = metrics.summarize(values)["success@5"]
    assert interval["low"] < 0.5 < interval["high"]
    same = metrics.paired(values["success@5"], values["success@5"])
    assert (same["estimate"], same["low"], same["high"]) == (0, 0, 0)
    better = metrics.paired(np.zeros(100), np.ones(100))
    assert better["low"] == 1.0 and better["better"] == 100


# --- Dense, fusion, rerank -----------------------------------------------------------

def test_reciprocal_rank_fusion_rewards_agreement():
    a = np.array([[1, 2, 3]])
    b = np.array([[3, 2, 4]])
    fused = dense.rrf([a, b], k=60, depth=4)
    assert list(fused[0][:2]) == [2, 3] or list(fused[0][:2]) == [3, 2]
    assert fused[0][0] in (2, 3) and set(fused[0]) == {1, 2, 3, 4}


def test_fusion_pads_short_lists():
    fused = dense.rrf([np.array([[1]]), np.array([[1]])], k=60, depth=3)
    assert list(fused[0]) == [1, -1, -1]


def test_small_corpora_pad_instead_of_failing(ds):
    model = experiments.fit_bm25(ds, bm25.Params())
    ranked, scores = model.search(["car"], 12)
    assert ranked.shape == (1, 12) and list(ranked[0][8:]) == [-1] * 4
    assert np.isneginf(scores[0][8:]).all()
    top, _ = dense.rerank(dense.OverlapScorer(), ["car"], ranked, ds.answers, depth=12)
    assert list(top[0][8:]) == [-1] * 4


def test_rerank_reorders_only_the_top(ds):
    candidates = np.array([[1, 2, 7, 3]])
    top, scores = dense.rerank(dense.OverlapScorer(), ["collision repair my car"],
                               candidates, ds.answers, depth=3)
    assert top[0][0] == 7 and set(top[0]) == {1, 2, 7}
    assert scores[0][0] >= scores[0][1]


def test_dense_index_caches_embeddings(ds, tmp_path):
    ids = ds.answer_ids
    docs = [ds.answers[i] for i in ids]
    index = dense.DenseIndex(dense.HashingEncoder(), tmp_path).fit(ids, docs)
    assert len(list(tmp_path.glob("*.npy"))) == 1
    again = dense.DenseIndex(dense.HashingEncoder(), tmp_path).fit(ids, docs)
    assert np.array_equal(index.vectors, again.vectors)
    ranked, _ = again.search([ds.answers[5]], 1)
    assert ranked[0][0] == 5


def test_every_real_model_declares_its_prefixes():
    assert set(dense.DENSE_MODELS) == set(dense.PREFIXES)
    assert dense.CURRENT_MODEL in dense.DENSE_MODELS


# --- Abstention ----------------------------------------------------------------------

def test_holding_out_a_domain_removes_only_its_answers(ds):
    removed, splits = experiments.gate_split(ds, "medicare-insurance")
    assert removed == {3, 4}
    answerable, unanswerable = splits["test"]
    assert [q.domain for q in unanswerable] == ["medicare-insurance"]
    assert all(q.domain != "medicare-insurance" for q in answerable)


def test_threshold_is_chosen_on_valid_and_applied_to_test():
    valid = (np.array([0.9, 0.8, 0.7]), np.array([0.2, 0.3]))
    test = (np.array([0.85, 0.1]), np.array([0.25, 0.75]))
    result = experiments.tuned_gate(valid, test, np.array([1.0, 0.0]))
    assert result["threshold"] == 0.7 and result["valid_balanced_accuracy"] == 1.0
    assert result["refused_when_answerable"] == 0.5
    assert result["refused_when_unanswerable"] == 0.5
    assert result["success@5_accepted_answerable"] == 1.0


def test_original_gate_uses_chroma_distance(ds):
    question = ds.split("test")[2]
    ranked = np.array([[7]])
    accepted = experiments.original_gate(ds, ranked, np.array([1.0]), [question])
    tokens = {"collision", "coverage", "repair", "car"}
    assert accepted[0] == accept_hybrid(0.0, 1.0) and tokens
    rejected = experiments.original_gate(ds, ranked, np.array([-0.5]), [question])
    assert not rejected[0]


# --- End to end ----------------------------------------------------------------------

def test_cpu_run_writes_results_and_report(ds, results, monkeypatch):
    params = experiments.tune_bm25(ds)
    assert isinstance(params, bm25.Params)
    result = experiments.run_bm25(ds)
    assert result["test_questions"] == 4
    gate = json.loads((results / "abstention_bm25.json").read_text())
    assert gate["held_out"]["domain"] in {"life-insurance", "medicare-insurance",
                                          "auto-insurance", "renters-insurance",
                                          "home-insurance"}
    experiments.write_result("dataset", data.summary(ds))
    assert report.DENSE_PENDING in report.document()
    assert "BM25: correct answer in top 5" in report.summary_table()


def test_dense_run_with_stand_ins_writes_everything(ds, results):
    experiments.tune_bm25(ds)
    encoders = {e.name: e for e in (dense.HashingEncoder("a", 64), dense.HashingEncoder("b", 128))}
    systems = experiments.run_dense(ds, encoders, dense.OverlapScorer(), cache_dir=None)
    assert {"bm25", "dense_current", "hybrid", "hybrid_rerank"} <= set(systems["systems"])
    assert "hybrid_rerank vs bm25" in systems["paired"]
    gate = json.loads((results / "abstention.json").read_text())
    assert {"original_gate", "tuned_cosine_gate", "tuned_rerank_gate"} <= set(gate)
    experiments.write_result("dataset", data.summary(ds))
    experiments.run_bm25(ds)
    document = report.document()
    assert report.DENSE_PENDING not in document and "Paired differences" in document


def test_report_check_catches_a_stale_readme(ds, results, tmp_path, monkeypatch):
    experiments.tune_bm25(ds)
    experiments.run_bm25(ds)
    experiments.write_result("dataset", data.summary(ds))
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.START}\nold\n{report.END}\nrest\n", encoding="utf-8")
    monkeypatch.setattr(report, "README", readme)
    monkeypatch.setattr(report, "DOC", tmp_path / "retrieval.md")
    assert report.check() == 1
    report.write()
    assert report.check() == 0
    assert readme.read_text(encoding="utf-8").endswith("rest\n")
