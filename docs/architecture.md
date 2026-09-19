# ClaimsIQ Architecture

ClaimsIQ is deliberately split into testable layers so the LLM is not responsible for data access or business logic.

```text
                Streamlit dashboard
                        │
                        ▼
                Groq tool-calling LLM
                        │
             allow-listed Python tools
             ┌──────────┼───────────┐
             ▼          ▼           ▼
          SQLite      Chroma      ML signals
             │          │           │
             ▼          ▼           ▼
        MTPL2 data   hybrid RAG   Isolation Forest
             │          │           │
             └──── materialised analytics caches ────┘
```

## Data layer

`src/data/mtpl2.py` downloads `freMTPL2freq` and `freMTPL2sev` from OpenML, validates identifiers/exposure/claim amounts, aggregates claim severity by policy and produces the modelling table.

## Analytics layer

`src/analytics/build_mtpl2_db.py` writes the prepared data to SQLite and creates indexes. `src/analytics/mtpl2_queries.py` contains reusable KPI, regional, fuel, driver-age and anomaly-diagnostic queries. The regional analysis includes a SQL `DENSE_RANK()` window function. `src/analytics/cache.py` materialises small aggregate tables so interactive tools do not repeatedly scan the full policy table.

## ML layer

`src/ml/mtpl2_anomaly.py` uses an Isolation Forest to generate **statistical review signals**. Log transforms reduce the effect of highly skewed claim/density variables, RobustScaler handles heavy-tailed numeric features, and geography (`Region`/`Area`) is excluded from model inputs so the model is not dominated by location alone. The resulting percentile score and review band are stored in SQLite.

This is an unsupervised prioritisation mechanism, not a fraud classifier.

## RAG layer

`src/rag/index_policy.py` embeds a small project-authored demonstration policy corpus using Sentence-Transformers and stores it in ChromaDB. `src/rag/retrieve.py` combines semantic similarity with lexical evidence and only accepts results that pass the hybrid relevance gate. These documents are **not insurer documents**.

## Agent layer

`src/agent/tools.py` exposes only read-only, allow-listed business operations. There is intentionally no arbitrary SQL tool. `src/agent/groq_agent.py` implements the local function-calling loop: the LLM selects a tool, ClaimsIQ executes it, the result is returned to the model, and the loop continues until the answer is ready or a fixed iteration limit is reached. This follows Groq's documented local-tool-calling pattern.

Independent tool calls are executed concurrently. Tool traces capture tool names, arguments and execution latency. Source metadata is retained separately so the UI can show which dataset or policy passage supported an answer.

## Evaluation layer

`eval/benchmark.json` defines representative business questions with expected tools. `src/evaluation/benchmark.py` measures tool exact-match rate, tool recall, source presence and end-to-end latency when a Groq API key is configured.

## UI

`app/streamlit_app.py` provides portfolio KPIs, readable regional/fuel/age views, anomaly diagnostics and the Copilot with answer/source/tool-trace sections.
