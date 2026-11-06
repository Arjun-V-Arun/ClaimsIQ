"""Command line for the retrieval benchmark. See src/retrieval/__init__.py."""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from src.retrieval import data, dense, experiments, report

SMOKE_QUESTIONS = 200


def build() -> None:
    ds = data.load(data.fetch())
    experiments.write_result("dataset", data.summary(ds))
    print("Tuning BM25 on the valid split (about two minutes)...")
    print(f"chosen: {experiments.tune_bm25(ds)}")


def evaluate(args) -> int:
    ds = data.load(data.fetch())
    if not args.dense:
        experiments.run_bm25(ds)
        report.write()
        return 0

    if args.smoke:
        # The full dense code path with tiny stand-in models, written to a
        # temporary folder so real results are never touched.
        out = Path(tempfile.mkdtemp(prefix="retrieval-smoke-"))
        shutil.copy(experiments.RESULTS_DIR / "selection_bm25.json", out)
        encoders = {e.name: e for e in (dense.HashingEncoder("hashing-a", 256),
                                        dense.HashingEncoder("hashing-b", 512))}
        experiments.run_dense(ds, encoders, dense.OverlapScorer(), cache_dir=None,
                              directory=out, device="CPU (stand-in models)",
                              limit=SMOKE_QUESTIONS)
        print(f"Smoke run passed. Output in {out} (not published).")
        return 0

    device = dense.device_name()
    if device == "CPU":
        print("No GPU found. This still works but encoding and reranking will be slow.")
    print(f"Loading models on {device}. First run downloads about 300 MB.")
    encoders = {name: dense.SentenceEncoder(name) for name in dense.DENSE_MODELS}
    experiments.run_dense(ds, encoders, dense.CrossEncoderScorer(), device=device)
    report.write()
    print("Done. Commit eval/retrieval/, docs/retrieval.md, and README.md.")
    return 0


def check() -> int:
    """CI: BM25 is deterministic, so the committed test numbers must reproduce."""
    ds = data.load(data.fetch())
    committed = experiments.read_result("bm25")["metrics"]
    model = experiments.fit_bm25(ds, experiments.bm25_params())
    test = ds.split("test")
    ranked, _ = model.search(experiments.texts(test), experiments.DEPTH)
    fresh = experiments.metrics.summarize(experiments.score(ranked, test),
                                          seed=experiments.SEED)
    drift = {name: (committed[name]["estimate"], fresh[name]["estimate"])
             for name in committed if committed[name] != fresh[name]}
    if drift:
        print("BM25 results changed. If intended, run `python -m src.retrieval eval` "
              f"and commit the new results. Changed (committed, now): {drift}")
        return 1
    print(f"BM25 reproduced exactly on {len(test)} test questions.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m src.retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    run = commands.add_parser("eval")
    run.add_argument("--dense", action="store_true", help="embedding, fusion, reranking")
    run.add_argument("--smoke", action="store_true", help="dense path with stand-in models")
    rep = commands.add_parser("report")
    rep.add_argument("--check", action="store_true")
    commands.add_parser("check")
    args = parser.parse_args()

    if args.command == "build":
        build()
    elif args.command == "eval":
        return evaluate(args)
    elif args.command == "report":
        return report.check() if args.check else (report.write() or 0)
    elif args.command == "check":
        return check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
