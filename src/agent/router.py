"""Deterministic fallback router for ClaimsIQ when no LLM key is configured."""

from __future__ import annotations

import re

from .tools import ClaimsTools


class ClaimsRouter:
    def __init__(self, tools: ClaimsTools) -> None:
        self.tools = tools

    def answer(self, question: str) -> dict:
        q = question.lower().strip()

        policy_match = re.search(r"\b(\d{4,8})\b", q)
        if policy_match and any(term in q for term in ["policy", "claim", "anomaly", "id"]):
            policy = self.tools.find_policy(int(policy_match.group(1)))
            if policy is None:
                return {"type": "policy_record", "answer": "No policy with that ID was found.", "sources": []}
            return {
                "type": "policy_record",
                "answer": f"Policy {policy['IDpol']} has {policy['ClaimNb']} observed claims, {policy['Exposure']:.2f} years of exposure, and an anomaly percentile of {policy.get('anomaly_score', 0):.3f}.",
                "details": policy,
                "sources": [{"kind": "dataset", "title": "French MTPL2 insurance dataset"}],
            }

        if any(term in q for term in ["covered", "coverage", "deductible", "eligible", "accidental damage", "theft"]):
            sources = self.tools.search_policy(question)
            return {
                "type": "policy",
                "answer": sources[0]["text"] if sources else "No relevant demonstration-policy passage found.",
                "sources": sources,
            }

        if "region" in q or "regional" in q:
            rows = self.tools.regional_summary()
            best = rows[0]
            return {
                "type": "analytics",
                "answer": f"{best['region']} has the highest observed claim frequency at {best['claims_per_exposure_year']:.5f} claims per exposure-year in this dataset.",
                "details": rows,
                "sources": [{"kind": "dataset", "title": "French MTPL2 insurance dataset"}],
            }

        if "fuel" in q or "diesel" in q or "regular" in q:
            rows = self.tools.fuel_summary()
            return {
                "type": "analytics",
                "answer": "The fuel breakdown is shown below.",
                "details": rows,
                "sources": [{"kind": "dataset", "title": "French MTPL2 insurance dataset"}],
            }

        if "driver" in q or "age" in q:
            rows = self.tools.driver_age_summary()
            return {
                "type": "analytics",
                "answer": "The driver-age breakdown is shown below.",
                "details": rows,
                "sources": [{"kind": "dataset", "title": "French MTPL2 insurance dataset"}],
            }

        if "anomal" in q or "unusual" in q or "review" in q:
            rows = self.tools.top_anomalies()
            return {
                "type": "analytics",
                "answer": "These are the highest statistical review signals. They are not confirmed fraud cases.",
                "details": rows,
                "sources": [{"kind": "dataset", "title": "French MTPL2 insurance dataset"}],
            }

        kpis = self.tools.get_kpis()
        return {
            "type": "analytics",
            "answer": f"The portfolio contains {int(kpis['policies']):,} policies and {int(kpis['total_claims']):,} observed claims, with {kpis['claims_per_exposure_year']:.5f} claims per exposure-year and average claim severity of €{kpis['avg_claim_severity_eur']:,.2f}.",
            "sources": [{"kind": "dataset", "title": "French MTPL2 insurance dataset"}],
        }
