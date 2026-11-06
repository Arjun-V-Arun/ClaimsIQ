"""Retrieval over real insurance questions, evaluated against outside labels.

The original copilot searched four project-written documents, so its perfect
retrieval scores could not tell good ranking from luck. This package replaces
that test with InsuranceQA: 27,413 answers written by insurance professionals
and 16,889 questions from real users, labelled by the dataset's authors.

    python -m src.retrieval build        download, verify, tune BM25 on valid
    python -m src.retrieval eval         BM25 on test (CPU, no downloads)
    python -m src.retrieval eval --dense dense, hybrid, and reranking (GPU)
    python -m src.retrieval report       regenerate docs/retrieval.md
    python -m src.retrieval check        CI: BM25 must reproduce exactly

Design and results: docs/retrieval.md.
"""
