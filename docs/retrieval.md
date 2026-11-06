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

Run on NVIDIA GeForce RTX 4070 Laptop GPU. Embedding model chosen on valid: `intfloat/e5-small-v2` (`all-MiniLM-L6-v2` 0.388, `bge-small-en-v1.5` 0.424, `e5-small-v2` 0.437 MRR@10). Fusion constant k = 10. Reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top 20 embedding candidates or the top 20 hybrid candidates.

Whole pipelines were also compared on valid, and the winner is the one this project would ship: Embeddings, best model on valid 0.437; Hybrid: BM25 + embeddings, rank fusion 0.410; Embeddings + cross-encoder rerank 0.456; Hybrid + cross-encoder rerank 0.456 MRR@10. Chosen: **Hybrid + cross-encoder rerank**.

| System | Success@1 | Success@5 (95% CI) | MRR@10 | nDCG@10 | Success@100 | Latency p50 / p95 |
|---|---|---|---|---|---|---|
| BM25 keyword search | 21.9% | 41.2% (39.1% to 43.4%) | 0.301 | 0.297 | 72.4% | 1 / 2 ms |
| Embeddings, the copilot's current model | 31.8% | 55.6% (53.4% to 57.7%) | 0.418 | 0.413 | 87.9% | 6 / 9 ms |
| Embeddings, best model on valid | 35.5% | 59.0% (57.0% to 61.0%) | 0.457 | 0.450 | 88.8% | 10 / 13 ms |
| Hybrid: BM25 + embeddings, rank fusion | 31.9% | 57.1% (55.0% to 59.2%) | 0.424 | 0.420 | 87.6% | 12 / 15 ms |
| Embeddings + cross-encoder rerank | 37.7% | 59.7% (57.6% to 61.8%) | 0.474 | 0.460 | 88.8% | 33 / 53 ms |
| Hybrid + cross-encoder rerank **(chosen on valid)** | 37.3% | 59.9% (57.8% to 62.1%) | 0.473 | 0.463 | 87.6% | 35 / 54 ms |

Paired differences on the same test questions, 95% intervals:

| Comparison | Success@5 | MRR@10 | Questions better / worse |
|---|---|---|---|
| Embeddings, the copilot's current model vs BM25 keyword search | +14.4 pts (+12.2 to +16.6) | +0.117 (+0.099 to +0.135) | 414 / 125 |
| Embeddings, best model on valid vs Embeddings, the copilot's current model | +3.4 pts (+1.7 to +5.0) | +0.038 (+0.025 to +0.052) | 190 / 123 |
| Hybrid: BM25 + embeddings, rank fusion vs Embeddings, best model on valid | -1.8 pts (-3.2 to -0.4) | -0.033 (-0.045 to -0.021) | 97 / 134 |
| Embeddings + cross-encoder rerank vs Embeddings, best model on valid | +0.7 pts (-0.8 to +2.1) | +0.017 (+0.005 to +0.029) | 123 / 109 |
| Hybrid + cross-encoder rerank vs Hybrid: BM25 + embeddings, rank fusion | +2.8 pts (+1.2 to +4.2) | +0.050 (+0.037 to +0.062) | 150 / 95 |
| Hybrid + cross-encoder rerank vs Embeddings + cross-encoder rerank | +0.2 pts (-0.5 to +1.0) | -0.001 (-0.004 to +0.003) | 35 / 31 |
| Hybrid + cross-encoder rerank vs BM25 keyword search | +18.7 pts (+16.8 to +20.8) | +0.172 (+0.155 to +0.188) | 442 / 68 |
| Hybrid + cross-encoder rerank vs Embeddings, best model on valid | +0.9 pts (-0.5 to +2.4) | +0.016 (+0.005 to +0.028) | 131 / 113 |

Success@5 by domain:

| Domain | Questions | BM25 | Chosen pipeline |
|---|---|---|---|
| life-insurance | 514 | 46.7% | 68.3% |
| medicare-insurance | 251 | 45.0% | 66.5% |
| auto-insurance | 249 | 33.3% | 53.4% |
| health-insurance | 245 | 42.0% | 59.6% |
| home-insurance | 201 | 30.3% | 46.8% |
| renters-insurance | 140 | 30.7% | 42.1% |
| disability-insurance | 102 | 47.1% | 67.7% |
| retirement-plans | 95 | 47.4% | 70.5% |
| annuities | 94 | 42.5% | 48.9% |
| long-term-care-insurance | 89 | 42.7% | 57.3% |
| critical-illness-insurance | 17 | 47.1% | 70.6% |
| other-insurance | 3 | 66.7% | 100.0% |

Questions the chosen pipeline missed entirely (no correct answer in its top 20), a seeded sample. Several retrieved answers may be correct but unlabelled, which is why every score here is a lower bound:

- **How Much Does Life Insurance Cost?** (life-insurance)
  - Retrieved first: Life insurance usually costs under $100 per month. The age of the Insured, the length of time that the price is locked in, the health of the person being insure...
  - A correct answer: Video Transcript: Today's question is, "How much does life insurance cost?" Well, it depends on your gender, your age, your smoking habits, your medical backgro...
- **Can Annuities Be Willed?** (annuities)
  - Retrieved first: Yes, annuities can be inherited. When the annuity is inherited during the annuitization stage, the recipients of the annuity payments will need to pay income ta...
  - A correct answer: Annuities are contracts. They identify a beneficiary to receive proceeds in the case of a premature death. The company is required to honor the beneficiary. If ...
- **What Is A Typical Home Insurance Cost?** (home-insurance)
  - Retrieved first: Great question. A home insurance policy cost on average about $300-$900 a year. It also depends on a multible factors such as your home location. The age of you...
  - A correct answer: Home insurance is written for homes. Homes are never typical. It is the uniqueness that makes home ownership attractive. A homeowners policy is based upon the v...
- **Can I Save Money On My Health Insurance Premiums In The Marketplace?** (health-insurance)
  - Retrieved first: I would work through an independent health insurance agent/broker who can review options for multiple companies and create a thorough comparison between plans. ...
  - A correct answer: That is an excellent question! You may very well, and it certainly doesn't hurt to look. There are a lot of things that will determine the cost of your plan, bu...
- **How Do Lifetime Annuities Work?** (annuities)
  - Retrieved first: Lifetime annuities are annuities that pay the annuitants monthly payments for the rest of their lives. The annuity can be an immediate annuity, where a lump sum...
  - A correct answer: The combination of mortality credits and the interest rate yield create a compelling reason to investigate lifetime income you cant outlive. Generally speaking,...
- **How Cheap Is Renters Insurance?** (renters-insurance)
  - Retrieved first: Renters insurance can be very affordable. The premiums for renter's insurance normally range between $100 and $300 depending on the limits selected for your par...
  - A correct answer: Renters Insurance can be purchased for as little as $8/mo.. Make sure to purchase a policy with replacement cost for your belongings. The coverage typically add...
- **What Happens When Your Homeowners Insurance Drops You?** (home-insurance)
  - Retrieved first: A car insurance company can drop your homeowner's insurance. It happens every day. For example, if your driving record was clean and you had no claims for 20 ye...
  - A correct answer: Well, if for some reason your company non-renews or cancelsyour Homeowners' coverage, the best thing to do would be to start researching coverage availablity wi...
- **I'm Pregnant Can I Get Health Insurance?** (health-insurance)
  - Retrieved first: Yes you can get health insurance when pregnant, but there are limitations. There are three places to turn. You can enroll in a group health plan through your em...
  - A correct answer: Yes. Under the new Affordable Care Act, you can not be declined for health insurance for any medical-related conditions. Each policy is required to cover 10 ess...

## Abstention

A retriever should refuse when the corpus has no answer. To test that with real questions, every answer belonging to **medicare-insurance** is removed from the index (2,869 answers), making its 251 test questions unanswerable, while 1,749 questions from other domains stay answerable. The domain was chosen by rule, on the train split only: eligible domain (100+ valid and test questions) whose train questions most often retrieve a same-domain answer first, by BM25. Thresholds are tuned on valid for balanced accuracy. A useful gate refuses unanswerable questions without refusing answerable ones, and the questions it accepts should score higher than all answerable questions do.

| Gate | Refused, unanswerable | Refused, answerable | Success@5, all answerable | Success@5, accepted |
|---|---|---|---|---|
| Copilot's original rule and thresholds | 31.1% | 0.9% | 55.1% | 55.1% |
| Tuned on top cosine, current model | 87.6% | 9.6% | 55.1% | 56.2% |
| Tuned on top cross-encoder score, chosen pipeline | 86.1% | 9.6% | 59.1% | 60.4% |
| Tuned on BM25 score per unit of query idf | 59.8% | 20.9% | 40.7% | 40.1% |

Removing a domain leaves related answers in other domains, so some "unanswerable" questions still have partly useful answers, and even a perfect gate would not refuse all of them.

## Revisions

- **First published run.** The pipeline always fused BM25 with embeddings before reranking. On test, fusion lowered accuracy, and the comparison a reader needs most, reranking against embeddings alone, was missing. Fusion and reranking are now chosen on valid like every other setting, embeddings + rerank is reported as its own system, and the chosen pipeline is compared with embeddings alone. The change was made after seeing test results, which is why it is recorded here. The choice itself uses only valid data.

## Limits

- One public dataset from one Q&A site, mostly about US insurance products.
- Labels mark the answers written for each question. Another answer can be correct without being labelled, so every score here is a lower bound.
- No model was fine-tuned on the training split. Doing so would likely raise every learned score.
- Latency is measured one query at a time on one machine and will differ elsewhere.
- The held-out domain was picked because it is the most self-contained, so its questions are the easiest kind to refuse. Questions close to an indexed topic but not answered by it would be harder.
- The copilot's `search_policy` tool still uses the four demonstration documents. Switching it to the winning retriever is the next step.

