"""Experiments. Each writes one JSON file to eval/retrieval/.

Every choice (BM25 settings, which embedding model, fusion constant, rerank
depth, abstention threshold) is made on the valid split by a rule fixed in
advance, MRR@10 unless stated. The test split is used only to report.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.rag.hybrid import accept_hybrid, content_tokens
from src.retrieval import bm25, data, dense, metrics
from src.retrieval.data import Dataset, Question

RESULTS_DIR = data.ROOT / "eval" / "retrieval"
CACHE_DIR = data.DATA_DIR / "embeddings"
SEED = 7
DEPTH = 100
SELECT_BY = "mrr@10"
BM25_GRID = {"k1": (0.9, 1.2, 1.5), "b": (0.4, 0.6, 0.75, 0.9), "stem": (False, True)}
RRF_K = (10, 30, 60)
RERANK_DEPTHS = (20, 50)
LATENCY_SAMPLE = 200
MIN_GATE_QUESTIONS = 100


def write_result(name: str, body: dict, directory: Path | None = None) -> Path:
    directory = directory or RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=data.ROOT).stdout.strip()
    meta = {"dataset": f"insuranceQA@{data.COMMIT[:12]}", "code_commit": commit,
            "generated": dt.date.today().isoformat(), "seed": SEED}
    path = directory / f"{name}.json"
    path.write_text(json.dumps({"meta": meta, **body}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.name}")
    return path


def read_result(name: str, directory: Path | None = None) -> dict:
    return json.loads(((directory or RESULTS_DIR) / f"{name}.json").read_text(encoding="utf-8"))


def texts(questions: list[Question]) -> list[str]:
    return [q.text for q in questions]


def gold(questions: list[Question]) -> list[tuple[int, ...]]:
    return [q.answers for q in questions]


def corpus(ds: Dataset, exclude: set[int] = frozenset()) -> tuple[list[int], list[str]]:
    ids = [i for i in ds.answer_ids if i not in exclude]
    return ids, [ds.answers[i] for i in ids]


def score(ranked: np.ndarray, questions: list[Question]) -> dict[str, np.ndarray]:
    return metrics.per_question(ranked, gold(questions))


# --- BM25 -------------------------------------------------------------------------

def fit_bm25(ds: Dataset, params: bm25.Params, exclude: set[int] = frozenset()) -> bm25.BM25:
    ids, docs = corpus(ds, exclude)
    return bm25.BM25(params).fit(ids, docs)


def tune_bm25(ds: Dataset, directory: Path | None = None) -> bm25.Params:
    valid = ds.split("valid")
    grid = []
    for stem in BM25_GRID["stem"]:
        model = fit_bm25(ds, bm25.Params(stem=stem))
        for k1 in BM25_GRID["k1"]:
            for b in BM25_GRID["b"]:
                model.reweight(k1, b)
                ranked, _ = model.search(texts(valid), 10)
                value = float(score(ranked, valid)[SELECT_BY].mean())
                grid.append({"k1": k1, "b": b, "stem": stem, SELECT_BY: round(value, 4)})
    best = max(grid, key=lambda row: row[SELECT_BY])
    chosen = bm25.Params(best["k1"], best["b"], best["stem"])
    write_result("selection_bm25", {"rule": f"highest {SELECT_BY} on valid",
                                    "chosen": asdict(chosen), "grid": grid}, directory)
    return chosen


def bm25_params(directory: Path | None = None) -> bm25.Params:
    return bm25.Params(**read_result("selection_bm25", directory)["chosen"])


# --- Shared reporting pieces ----------------------------------------------------------

def by_domain(scores: dict[str, np.ndarray], questions: list[Question]) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(questions):
        groups[q.domain].append(i)
    return {domain: {"n": len(rows),
                     "success@5": round(float(scores["success@5"][rows].mean()), 4)}
            for domain, rows in sorted(groups.items(), key=lambda kv: -len(kv[1]))}


def latency(run_one, questions: list[Question]) -> dict:
    """Per-query wall time, one query at a time, as an interactive caller sees it."""
    rng = random.Random(SEED)
    sample = rng.sample(questions, min(LATENCY_SAMPLE, len(questions)))
    run_one(sample[0].text)                      # warm-up, not counted
    seconds: list[float] = []
    for q in sample:
        with dense.Timer(seconds):
            run_one(q.text)
    ms = np.array(seconds) * 1000
    return {"queries": len(sample), "p50_ms": round(float(np.percentile(ms, 50)), 1),
            "p95_ms": round(float(np.percentile(ms, 95)), 1)}


def keyword_pool(questions: list[Question], ranked_bm25: np.ndarray) -> dict:
    """How often the dataset's own 100-answer keyword pools hold the right answers."""
    anyone = np.mean([bool(set(q.answers) & set(q.pool)) for q in questions])
    every = np.mean([set(q.answers) <= set(q.pool) for q in questions])
    ours = score(ranked_bm25, questions)
    return {"pool_size": len(questions[0].pool),
            "solr_pool_has_a_correct_answer": round(float(anyone), 4),
            "solr_pool_has_every_correct_answer": round(float(every), 4),
            "our_bm25_top100_has_a_correct_answer": round(float(ours["success@100"].mean()), 4),
            "our_bm25_top100_has_every_correct_answer":
                round(float((ours["recall@100"] == 1).mean()), 4)}


def failures(ranked: np.ndarray, questions: list[Question], answers: dict[int, str],
             count: int = 8) -> list[dict]:
    misses = [i for i, q in enumerate(questions)
              if ranked[i, 0] >= 0 and not set(q.answers) & set(int(a) for a in ranked[i, :20])]
    rng = random.Random(SEED)
    picked = sorted(rng.sample(misses, min(count, len(misses))))
    clip = lambda t: t[:160] + ("..." if len(t) > 160 else "")
    return [{"qid": questions[i].qid, "domain": questions[i].domain,
             "question": questions[i].text,
             "retrieved_first": clip(answers[int(ranked[i, 0])]),
             "a_correct_answer": clip(answers[questions[i].answers[0]])} for i in picked]


# --- Abstention ------------------------------------------------------------------------

def answer_domains(ds: Dataset) -> dict[int, str]:
    owners: dict[int, Counter] = defaultdict(Counter)
    for rows in ds.questions.values():
        for q in rows:
            for a in q.answers:
                owners[a][q.domain] += 1
    return {a: c.most_common(1)[0][0] for a, c in owners.items()}


def choose_held_out(ds: Dataset, params: bm25.Params) -> dict:
    """Pick the domain to remove, using only the train split.

    The cleanest unanswerable questions come from the domain whose questions
    most reliably retrieve answers from that same domain, so removing it leaves
    the fewest acceptable answers behind.
    """
    owner = answer_domains(ds)
    model = fit_bm25(ds, params)
    train = ds.split("train")
    ranked, _ = model.search(texts(train), 1)
    same: dict[str, list[bool]] = defaultdict(list)
    for q, top in zip(train, ranked[:, 0]):
        same[q.domain].append(owner[int(top)] == q.domain)
    sizes = {s: Counter(q.domain for q in ds.split(s)) for s in ("valid", "test")}
    eligible = {d: float(np.mean(v)) for d, v in same.items()
                if sizes["valid"][d] >= MIN_GATE_QUESTIONS and sizes["test"][d] >= MIN_GATE_QUESTIONS}
    domain = max(eligible, key=eligible.get)
    return {"domain": domain,
            "rule": "eligible domain (100+ valid and test questions) whose train questions "
                    "most often retrieve a same-domain answer first, by BM25",
            "same_domain_rate": {d: round(r, 3) for d, r in
                                 sorted(eligible.items(), key=lambda kv: -kv[1])}}


def gate_split(ds: Dataset, domain: str) -> tuple[set[int], dict[str, tuple[list, list]]]:
    owner = answer_domains(ds)
    removed = {a for a, d in owner.items() if d == domain}
    splits = {}
    for split in ("valid", "test"):
        rows = ds.split(split)
        answerable = [q for q in rows if q.domain != domain and not set(q.answers) & removed]
        unanswerable = [q for q in rows if q.domain == domain]
        splits[split] = (answerable, unanswerable)
    return removed, splits


def _rates(accept_answerable: np.ndarray, accept_unanswerable: np.ndarray,
           success_answerable: np.ndarray) -> dict:
    kept = accept_answerable.astype(bool)
    return {
        "refused_when_unanswerable": round(float(1 - accept_unanswerable.mean()), 4),
        "refused_when_answerable": round(float(1 - accept_answerable.mean()), 4),
        "success@5_all_answerable": round(float(success_answerable.mean()), 4),
        "success@5_accepted_answerable": round(float(success_answerable[kept].mean()), 4)
        if kept.any() else None,
    }


def tuned_gate(conf_valid: tuple[np.ndarray, np.ndarray],
               conf_test: tuple[np.ndarray, np.ndarray],
               success_test: np.ndarray) -> dict:
    """Threshold chosen on valid to maximize balanced accuracy, applied to test."""
    answerable, unanswerable = conf_valid
    candidates = np.unique(np.concatenate([answerable, unanswerable]))
    def balanced(t):
        return 0.5 * ((answerable >= t).mean() + (unanswerable < t).mean())
    threshold = float(max(candidates, key=balanced))
    test_answerable, test_unanswerable = conf_test
    return {"threshold": round(threshold, 4),
            "valid_balanced_accuracy": round(float(balanced(threshold)), 4),
            **_rates(test_answerable >= threshold, test_unanswerable >= threshold, success_test)}


def bm25_confidence(model: bm25.BM25, questions: list[Question]) -> tuple[np.ndarray, np.ndarray]:
    ranked, scores = model.search(texts(questions), 5)
    weight = model.query_weight(texts(questions))
    return ranked, np.divide(scores[:, 0], weight, out=np.zeros(len(questions)), where=weight > 0)


def original_gate(ds: Dataset, ranked: np.ndarray, top_cosine: np.ndarray,
                  questions: list[Question]) -> np.ndarray:
    """The copilot's existing accept rule, with its original thresholds.

    Chroma reports squared L2 distance, which for unit vectors is 2 - 2 cos.
    """
    accepted = []
    for i, q in enumerate(questions):
        if ranked[i, 0] < 0:
            accepted.append(False)
            continue
        query_tokens = content_tokens(q.text)
        doc_tokens = content_tokens(ds.answers[int(ranked[i, 0])])
        coverage = len(query_tokens & doc_tokens) / len(query_tokens) if query_tokens else 0.0
        accepted.append(accept_hybrid(2 - 2 * float(top_cosine[i]), coverage))
    return np.array(accepted)


# --- CPU run: BM25 ---------------------------------------------------------------------

def run_bm25(ds: Dataset, directory: Path | None = None) -> dict:
    params = bm25_params(directory)
    model = fit_bm25(ds, params)
    test = ds.split("test")
    ranked, _ = model.search(texts(test), DEPTH)
    scores = score(ranked, test)
    result = {
        "system": "bm25", "params": asdict(params), "test_questions": len(test),
        "metrics": metrics.summarize(scores, seed=SEED),
        "by_domain": by_domain(scores, test),
        "latency": {**latency(lambda q: model.search([q], DEPTH), test), "device": "CPU"},
        "keyword_pools": keyword_pool(test, ranked),
    }
    write_result("bm25", result, directory)

    held = choose_held_out(ds, params)
    removed, splits = gate_split(ds, held["domain"])
    gated = fit_bm25(ds, params, removed)
    confidence, success = {}, None
    for split, (answerable, unanswerable) in splits.items():
        ranked_a, conf_a = bm25_confidence(gated, answerable)
        _, conf_u = bm25_confidence(gated, unanswerable)
        confidence[split] = (conf_a, conf_u)
        if split == "test":
            success = score(ranked_a, answerable)["success@5"]
    write_result("abstention_bm25", {
        "held_out": held, "answers_removed": len(removed),
        "questions": {s: {"answerable": len(a), "unanswerable": len(u)}
                      for s, (a, u) in splits.items()},
        "bm25": {"confidence": "top-1 score divided by the query's total idf",
                 **tuned_gate(confidence["valid"], confidence["test"], success)},
    }, directory)
    return result


# --- GPU run: dense, fusion, reranking -------------------------------------------------

@dataclass(frozen=True)
class Choice:
    """Everything chosen on valid that defines a pipeline."""
    best: str                 # embedding model with the highest valid score
    current: str              # the copilot's existing model, for comparison
    k_rrf: int
    depths: dict              # rerank depth per candidate generator: dense, hybrid


RERANKED = {"dense_rerank": "dense", "hybrid_rerank": "hybrid"}


class Pipeline:
    """Holds fitted retrievers so each experiment reuses the same indexes."""

    def __init__(self, ds: Dataset, encoders: dict[str, dense.Encoder], scorer: dense.Scorer,
                 cache_dir: Path | None, directory: Path | None):
        self.ds = ds
        self.scorer = scorer
        self.directory = directory
        self.bm25 = fit_bm25(ds, bm25_params(directory))
        ids, docs = corpus(ds)
        self.dense = {name: dense.DenseIndex(enc, cache_dir).fit(ids, docs)
                      for name, enc in encoders.items()}

    def without(self, removed: set[int], directory: Path | None) -> "Pipeline":
        """The same retrievers over a smaller corpus. Embeddings are reused."""
        gated = Pipeline.__new__(Pipeline)
        gated.ds, gated.scorer, gated.directory = self.ds, self.scorer, directory
        gated.bm25 = fit_bm25(self.ds, bm25_params(directory), removed)
        keep = np.array([i not in removed for i in self.bm25.ids])
        gated.dense = {}
        for name, index in self.dense.items():
            sub = dense.DenseIndex(index.encoder)
            sub.ids, sub.vectors = index.ids[keep], index.vectors[keep]
            gated.dense[name] = sub
        return gated

    def hybrid(self, model: str, questions: list[Question], k: int) -> np.ndarray:
        keyword, _ = self.bm25.search(texts(questions), DEPTH)
        semantic, _ = self.dense[model].search(texts(questions), DEPTH)
        return dense.rrf([keyword, semantic], k, DEPTH)

    def candidates(self, generator: str, questions: list[Question], choice: Choice) -> np.ndarray:
        if generator == "hybrid":
            return self.hybrid(choice.best, questions, choice.k_rrf)
        return self.dense[choice.best].search(texts(questions), DEPTH)[0]

    def reranked(self, candidates: np.ndarray, questions: list[Question], depth: int):
        top, top_scores = dense.rerank(self.scorer, texts(questions), candidates,
                                       self.ds.answers, depth)
        return np.concatenate([top, candidates[:, depth:]], axis=1), top_scores

    def run(self, name: str, questions: list[Question], choice: Choice) -> np.ndarray:
        if name == "bm25":
            return self.bm25.search(texts(questions), DEPTH)[0]
        if name == "dense_current":
            return self.dense[choice.current].search(texts(questions), DEPTH)[0]
        if name == "dense_best":
            return self.dense[choice.best].search(texts(questions), DEPTH)[0]
        if name == "hybrid":
            return self.candidates("hybrid", questions, choice)
        generator = RERANKED[name]
        return self.reranked(self.candidates(generator, questions, choice), questions,
                             choice.depths[generator])[0]


def run_dense(ds: Dataset, encoders: dict[str, dense.Encoder], scorer: dense.Scorer,
              cache_dir: Path | None = CACHE_DIR, directory: Path | None = None,
              device: str = "CPU", limit: int | None = None) -> dict:
    if limit:   # smoke runs: same code, fewer questions
        ds = Dataset(ds.answers, {s: rows[:limit] for s, rows in ds.questions.items()})
    pipe = Pipeline(ds, encoders, scorer, cache_dir, directory)
    valid, test = ds.split("valid"), ds.split("test")
    mean = lambda ranked, qs: float(score(ranked, qs)[SELECT_BY].mean())

    # 1. Choose on valid. Whether to fuse with BM25 and whether to rerank are
    #    choices like any other, so the shipped pipeline is the valid winner.
    dense_valid = {name: mean(index.search(texts(valid), 10)[0], valid)
                   for name, index in pipe.dense.items()}
    best = max(dense_valid, key=dense_valid.get)
    current = dense.CURRENT_MODEL if dense.CURRENT_MODEL in pipe.dense else next(iter(pipe.dense))
    rrf_valid = {k: mean(pipe.hybrid(best, valid, k), valid) for k in RRF_K}
    k_rrf = max(rrf_valid, key=rrf_valid.get)
    draft = Choice(best, current, k_rrf, {})
    rerank_valid = {
        generator: {d: mean(pipe.reranked(found, valid, d)[0], valid) for d in RERANK_DEPTHS}
        for generator, found in ((g, pipe.candidates(g, valid, draft)) for g in ("dense", "hybrid"))
    }
    depths = {g: max(v, key=v.get) for g, v in rerank_valid.items()}
    choice = Choice(best, current, k_rrf, depths)
    pipelines_valid = {"dense_best": dense_valid[best], "hybrid": rrf_valid[k_rrf],
                       "dense_rerank": rerank_valid["dense"][depths["dense"]],
                       "hybrid_rerank": rerank_valid["hybrid"][depths["hybrid"]]}
    chosen = max(pipelines_valid, key=pipelines_valid.get)
    write_result("selection_dense", {
        "rule": f"highest {SELECT_BY} on valid at each step, and for the whole pipeline",
        "dense_models": {k: round(v, 4) for k, v in dense_valid.items()},
        "chosen_model": best,
        "rrf_k": {str(k): round(v, 4) for k, v in rrf_valid.items()}, "chosen_rrf_k": k_rrf,
        "rerank_depth": {g: {str(d): round(v, 4) for d, v in r.items()}
                         for g, r in rerank_valid.items()},
        "chosen_rerank_depth": depths, "reranker": scorer.name,
        "pipelines_valid": {k: round(v, 4) for k, v in pipelines_valid.items()},
        "chosen_pipeline": chosen,
        "encode_seconds": {n: round(getattr(i, "encode_seconds", 0.0), 1)
                           for n, i in pipe.dense.items()},
    }, directory)

    # 2. Score every system once on test.
    names = ["bm25", "dense_current"] + (["dense_best"] if best != current else []) + \
        ["hybrid", "dense_rerank", "hybrid_rerank"]
    alias = lambda name: "dense_current" if name == "dense_best" and best == current else name
    chosen = alias(chosen)
    embeddings = alias("dense_best")
    rankings = {name: pipe.run(name, test, choice) for name in names}
    scores = {name: score(r, test) for name, r in rankings.items()}

    def one(name):
        return lambda q: pipe.run(name, [Question("x", "x", "x", q, (0,), ())], choice)

    pairs = [("bm25", "dense_current"), ("dense_current", "dense_best"),
             (embeddings, "hybrid"), (embeddings, "dense_rerank"), ("hybrid", "hybrid_rerank"),
             ("dense_rerank", "hybrid_rerank"), ("bm25", chosen), (embeddings, chosen)]
    seen, steps = set(), []
    for a, b in pairs:
        if a in scores and b in scores and a != b and (a, b) not in seen:
            seen.add((a, b))
            steps.append((a, b))

    systems = {
        "test_questions": len(test), "device": device, "chosen": chosen,
        "embeddings_alone": embeddings,
        "models": {"dense_current": current, "dense_best": best, "reranker": scorer.name,
                   "rrf_k": k_rrf, "rerank_depth": depths},
        "systems": {name: {"metrics": metrics.summarize(s, seed=SEED),
                           "latency": latency(one(name), test)}
                    for name, s in scores.items()},
        "paired": {f"{b} vs {a}": {m: metrics.paired(scores[a][m], scores[b][m], seed=SEED)
                                   for m in metrics.HEADLINE} for a, b in steps},
        "by_domain": {name: by_domain(scores[name], test) for name in ("bm25", chosen)},
        "failures": failures(rankings[chosen], test, ds.answers),
    }
    write_result("systems", systems, directory)

    # 3. Abstention: the copilot's original rule, a tuned cosine threshold, and a
    #    tuned threshold on the cross-encoder score over the chosen pipeline's
    #    candidates (the hybrid ones if the chosen pipeline fuses).
    held = choose_held_out(ds, bm25_params(directory)) if not limit else \
        {"domain": Counter(q.domain for q in test).most_common(1)[0][0], "rule": "smoke run"}
    removed, splits = gate_split(ds, held["domain"])
    gated = pipe.without(removed, directory)
    generator = "hybrid" if chosen in ("hybrid", "hybrid_rerank") else "dense"

    rerank_conf, cosine_conf, original = {}, {}, {}
    success_chosen = success_current = None
    for split, (answerable, unanswerable) in splits.items():
        pair_rerank, pair_cosine, pair_original = [], [], []
        for group in (answerable, unanswerable):
            found = gated.candidates(generator, group, choice)
            _, top_scores = gated.reranked(found, group, depths[generator])
            pair_rerank.append(top_scores[:, 0])
            current_ranked, current_scores = gated.dense[current].search(texts(group), 1)
            pair_cosine.append(current_scores[:, 0])
            pair_original.append(original_gate(ds, current_ranked, current_scores[:, 0], group))
            if split == "test" and group is answerable:
                success_chosen = score(gated.run(chosen, group, choice), group)["success@5"]
                success_current = score(gated.run("dense_current", group, choice),
                                        group)["success@5"]
        rerank_conf[split], cosine_conf[split], original[split] = (
            tuple(pair_rerank), tuple(pair_cosine), tuple(pair_original))

    write_result("abstention", {
        "held_out": held, "answers_removed": len(removed),
        "questions": {s: {"answerable": len(a), "unanswerable": len(u)}
                      for s, (a, u) in splits.items()},
        "original_gate": {
            "system": f"dense_current ({current}) with the copilot's original thresholds",
            **_rates(*original["test"], success_current)},
        "tuned_rerank_gate": {
            "system": f"{chosen}, threshold on the top cross-encoder score over "
                      f"{generator} candidates",
            **tuned_gate(rerank_conf["valid"], rerank_conf["test"], success_chosen)},
        "tuned_cosine_gate": {
            "system": f"dense_current ({current}), threshold on top cosine",
            **tuned_gate(cosine_conf["valid"], cosine_conf["test"], success_current)},
    }, directory)
    return systems
