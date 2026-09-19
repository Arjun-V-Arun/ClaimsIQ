"""Streamlit dashboard for ClaimsIQ."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.agent.router import ClaimsRouter
from src.agent.tools import ClaimsTools
from src.analytics.mtpl2_queries import (
    anomaly_region_summary,
    driver_age_summary,
    fuel_summary,
    kpis,
    regional_summary,
    top_anomalies,
)

DB_PATH = ROOT / "data" / "claims_real.db"
POLICY_DB_PATH = ROOT / "data" / "chroma"

st.set_page_config(page_title="ClaimsIQ", page_icon="📊", layout="wide")

st.title("ClaimsIQ")
st.caption("Real-world motor insurance analytics + GenAI copilot")

if not DB_PATH.exists():
    st.warning("Real dataset not found. Run `python scripts/bootstrap_real.py` from the project root first.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_tools() -> ClaimsTools:
    return ClaimsTools(DB_PATH, policy_dir=POLICY_DB_PATH)


@st.cache_resource(show_spinner=False)
def get_agent(tools: ClaimsTools):
    if not os.environ.get("GROQ_API_KEY"):
        return None
    from src.agent.groq_agent import ClaimsAgent
    return ClaimsAgent(tools)

tools = get_tools()
kpi = kpis(DB_PATH).iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Policies", f"{int(kpi['policies']):,}")
col2.metric("Policies with claims", f"{int(kpi['policies_with_claims']):,}")
col3.metric("Total claims", f"{int(kpi['total_claims']):,}")
col4.metric("Claims / exposure-year", f"{kpi['claims_per_exposure_year']:.4f}")
col5.metric("Avg. claim severity", f"€{kpi['avg_claim_severity_eur']:,.2f}")

st.subheader("Regional performance")
regional = regional_summary(DB_PATH)
st.dataframe(
    regional.style.format({
        "policy_count": "{:,.0f}",
        "policies_with_claims": "{:,.0f}",
        "claim_count": "{:,.0f}",
        "exposure_years": "{:,.2f}",
        "pct_policies_with_claim": "{:.2f}",
        "claims_per_exposure_year": "{:.5f}",
        "total_claim_amount_eur": "€{:,.2f}",
        "avg_claim_severity_eur": "€{:,.2f}",
        "loss_cost_per_exposure_year_eur": "€{:,.2f}",
    }),
    width="stretch",
    hide_index=True,
)

left, right = st.columns(2)
with left:
    st.subheader("Claim frequency by region")
    freq_plot = regional.sort_values("claims_per_exposure_year", ascending=True).tail(10)
    st.bar_chart(freq_plot.set_index("region")["claims_per_exposure_year"])
    st.caption("Top 10 regions by observed claim frequency, measured per exposure-year.")

with right:
    st.subheader("Claim severity by fuel type")
    fuel = fuel_summary(DB_PATH)
    st.bar_chart(fuel.set_index("fuel_type")["avg_claim_severity_eur"])
    st.caption("Average observed claim amount among policies with claims.")

st.subheader("Driver age profile")
age = driver_age_summary(DB_PATH)
st.dataframe(age, width="stretch", hide_index=True)

st.subheader("Statistical review signals")
st.caption(
    "Isolation Forest prioritises unusual records for review. The model excludes Region/Area "
    "from its feature inputs to reduce geography-driven concentration. Percentile is empirical "
    "extremeness within this dataset; these are not confirmed fraud cases."
)

anom_left, anom_right = st.columns([2, 1])
with anom_left:
    anomalies = top_anomalies(DB_PATH, 10)
    display_cols = [
        "IDpol", "ClaimNb", "Exposure", "DrivAge", "VehAge", "BonusMalus",
        "Region", "anomaly_percentile_pct", "review_band",
    ]
    available = [c for c in display_cols if c in anomalies.columns]
    st.dataframe(
        anomalies[available].style.format({"anomaly_percentile_pct": "{:.2f}%"}),
        width="stretch",
        hide_index=True,
    )
with anom_right:
    st.write("**Flag concentration diagnostic**")
    region_diag = anomaly_region_summary(DB_PATH)
    st.dataframe(
        region_diag[[
            "region", "flag_rate_pct", "portfolio_share_pct",
            "flag_share_pct", "flag_rate_vs_portfolio_ratio",
        ]],
        width="stretch",
        hide_index=True,
    )
    st.caption("A ratio near 1.0 means the region's share of flags is close to its portfolio share.")

st.divider()
st.subheader("ClaimsIQ Copilot")
st.caption("Ask questions about the observed portfolio or the demonstration policy corpus.")

example_questions = [
    "Which region has the highest claim frequency?",
    "Compare claim severity for Diesel and Regular vehicles.",
    "What is the claim frequency across driver age bands?",
    "Show me the most unusual policies flagged by the anomaly detector.",
    "Is accidental screen damage covered?",
]
selected = st.selectbox("Example questions", ["Custom question"] + example_questions)
question = st.text_input(
    "Ask a business question",
    value="" if selected == "Custom question" else selected,
    placeholder="e.g. Which region has the highest claim frequency?",
)

if question:
    try:
        if os.environ.get("GROQ_API_KEY"):
            agent = get_agent(tools)
            with st.spinner("ClaimsIQ is selecting tools and analysing the data..."):
                result = agent.answer(question)
            st.markdown(result["answer"])

            if result.get("sources"):
                st.write("**Sources**")
                for source in result["sources"]:
                    if source.get("kind") == "dataset":
                        st.caption(f"Dataset: {source['title']} ({source['detail']})")
                    elif source.get("kind") == "policy":
                        distance = source.get("distance")
                        lexical = source.get("lexical_coverage")
                        hybrid = source.get("hybrid_score")
                        suffix_parts = []
                        if isinstance(distance, (float, int)):
                            suffix_parts.append(f"distance={distance:.3f}")
                        if isinstance(lexical, (float, int)):
                            suffix_parts.append(f"lexical={lexical:.0%}")
                        if isinstance(hybrid, (float, int)):
                            suffix_parts.append(f"hybrid={hybrid:.3f}")
                        suffix = " | " + ", ".join(suffix_parts) if suffix_parts else ""
                        st.caption(f"Policy document: {source['title']}{suffix}")

            metrics_cols = st.columns(3)
            metrics_cols[0].metric("LLM calls", result.get("llm_calls", 0))
            metrics_cols[1].metric("LLM time", f"{result.get('llm_total_ms', 0.0) / 1000:.1f}s")
            metrics_cols[2].metric(
                "Tool time",
                f"{sum(float(t.get('latency_ms') or 0) for t in result.get('tool_trace', [])) / 1000:.2f}s",
            )


            if result.get("tool_trace"):
                with st.expander("Agent trace"):
                    st.dataframe(
                        pd.DataFrame(result["tool_trace"]),
                        width="stretch",
                        hide_index=True,
                    )
        else:
            st.info("No GROQ_API_KEY found, so ClaimsIQ is using its deterministic fallback router.")
            result = ClaimsRouter(tools).answer(question)
            st.markdown(result["answer"])
            if result.get("details"):
                st.dataframe(pd.DataFrame(result["details"]), width="stretch", hide_index=True)
            for source in result.get("sources", []):
                st.caption(f"Source: {source.get('title', 'Unknown source')}")
    except Exception as exc:
        st.error(f"Copilot error: {exc}")

st.divider()
st.caption(
    "Dataset: freMTPL2freq + freMTPL2sev via OpenML/CASdatasets. "
    "Public research data; not Assurant data. Policy corpus is a project demonstration corpus."
)
