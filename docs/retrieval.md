# Retrieval Benchmark

Generated from `eval/retrieval/` by `python -m src.retrieval report`. Do not edit by hand.

Every setting below was chosen on the valid split by a rule fixed in advance (highest MRR@10 unless stated). The test split was used only to report.

## Data

- 27,413 answers written by insurance professionals, median 78 words.
- 16,889 questions from real users: 12,889 train, 2,000 valid, 2,000 test, as published.
- 6,666 questions have more than one correct answer, so two families of metric are kept apart. Success@k: is any correct answer in the top k? Recall@k: what share of them are?
- 12 domains. Source: github.com/shuzi/insuranceQA@5c380dd08606 (V2, raw text). Released for research use only, so it is downloaded at build time and not committed. Feng et al., ASRU 2015.

## Keyword Baseline

BM25 with k1 = 0.9, b = 0.75, stemming on, chosen from 24 settings.

| Metric | Test (95% CI) |
|---|---|
| Success@1 | 21.9% (20.2% to 23.8%) |
| Success@5 | 41.2% (39.1% to 43.4%) |
| Success@20 | 57.5% (55.2% to 59.6%) |
| Success@100 | 72.4% (70.5% to 74.4%) |
| Recall@20 | 47.2% (45.2% to 49.2%) |
| Recall@100 | 63.7% (61.8% to 65.7%) |
| MRR@10 | 0.301 (0.283 to 0.320) |
| nDCG@10 | 0.297 (0.282 to 0.315) |

Latency: 2.3 ms median, 2.7 ms p95, one query at a time on CPU.

The dataset ships a 100-answer candidate pool per question from keyword search. It contains a correct answer for 66.1% of test questions and every correct answer for 48.1%. This project's BM25 top 100 reaches 72.4% and 54.5%. Keyword search alone misses many correct answers entirely, which is the gap learned retrieval has to close.

## Learned Retrieval

*Not run yet. Embedding, fusion, and reranking results come from `python -m src.retrieval eval --dense` on a machine that can download models.*

## Abstention

A retriever should refuse when the corpus has no answer. To test that with real questions, every answer belonging to **medicare-insurance** is removed from the index (2,869 answers), making its 251 test questions unanswerable, while 1,749 questions from other domains stay answerable. The domain was chosen by rule, on the train split only: eligible domain (100+ valid and test questions) whose train questions most often retrieve a same-domain answer first, by BM25. Thresholds are tuned on valid for balanced accuracy. A useful gate refuses unanswerable questions without refusing answerable ones, and the questions it accepts should score higher than all answerable questions do.

| Gate | Refused, unanswerable | Refused, answerable | Success@5, all answerable | Success@5, accepted |
|---|---|---|---|---|
| Tuned on BM25 score per unit of query idf | 59.8% | 20.9% | 40.7% | 40.1% |

Removing a domain leaves related answers in other domains, so some "unanswerable" questions still have partly useful answers, and even a perfect gate would not refuse all of them.

## Limits

- One public dataset from one Q&A site, mostly about US insurance products.
- Labels mark the answers written for each question. Another answer can be correct without being labelled, so every score here is a lower bound.
- No model was fine-tuned on the training split. Doing so would likely raise every learned score.
- Latency is measured one query at a time on one machine and will differ elsewhere.
- The copilot's `search_policy` tool still uses the four demonstration documents. Switching it to the winning retriever is the next step.

