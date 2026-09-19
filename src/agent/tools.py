"""Controlled, read-only business tools for ClaimsIQ."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from src.analytics.cache import CACHE_TABLES
from src.analytics.mtpl2_queries import (
    DRIVER_AGE_SQL,
    FUEL_SQL,
    KPI_SQL,
    REGIONAL_SQL,
    anomaly_region_summary,
    find_policy,
    run_query,
    top_anomalies,
)

if TYPE_CHECKING:
    from src.rag.retrieve import PolicyRetriever


DATASET_SOURCE = {
    "kind": "dataset",
    "title": "French MTPL2 insurance dataset",
    "detail": "Public freMTPL2freq + freMTPL2sev data; not Assurant data",
}


class ClaimsTools:
    def __init__(self, db_path: Path, policy_dir: Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.policy_dir = (
            Path(policy_dir) if policy_dir is not None else self.db_path.parent / "chroma"
        )
        self._retriever: "PolicyRetriever | None" = None
        with sqlite3.connect(self.db_path) as conn:
            self._tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

    @property
    def retriever(self) -> "PolicyRetriever":
        if self._retriever is None:
            from src.rag.retrieve import PolicyRetriever
            self._retriever = PolicyRetriever(self.policy_dir)
        return self._retriever

    def warmup_rag(self) -> None:
        """Load the embedding model/vector store before measured query latency."""
        _ = self.retriever.collection.count()

    def _has_table(self, table_name: str) -> bool:
        return table_name in self._tables

    def get_kpis(self) -> dict:
        if self._has_table(CACHE_TABLES["kpis"]):
            return run_query(self.db_path, f"SELECT * FROM {CACHE_TABLES['kpis']}").iloc[0].to_dict()
        return run_query(self.db_path, KPI_SQL).iloc[0].to_dict()

    def regional_summary(self) -> list[dict]:
        df = (
            run_query(self.db_path, f"SELECT * FROM {CACHE_TABLES['regional']}")
            if self._has_table(CACHE_TABLES["regional"])
            else run_query(self.db_path, REGIONAL_SQL)
        )
        columns = [
            "region", "policy_count", "claim_count",
            "claims_per_exposure_year", "avg_claim_severity_eur",
            "loss_cost_per_exposure_year_eur", "frequency_rank"
        ]
        return df[columns].to_dict(orient="records")

    def fuel_summary(self) -> list[dict]:
        df = (
            run_query(self.db_path, f"SELECT * FROM {CACHE_TABLES['fuel']}")
            if self._has_table(CACHE_TABLES["fuel"])
            else run_query(self.db_path, FUEL_SQL)
        )
        columns = [
            "fuel_type", "policy_count", "claim_count",
            "claims_per_exposure_year", "avg_claim_severity_eur",
            "total_claim_amount_eur"
        ]
        return df[columns].to_dict(orient="records")

    def driver_age_summary(self) -> list[dict]:
        df = (
            run_query(self.db_path, f"SELECT * FROM {CACHE_TABLES['age']}")
            if self._has_table(CACHE_TABLES["age"])
            else run_query(self.db_path, DRIVER_AGE_SQL)
        )
        columns = [
            "driver_age_band", "policy_count", "claim_count",
            "claims_per_exposure_year", "avg_claim_severity_eur"
        ]
        return df[columns].to_dict(orient="records")

    def find_policy(self, policy_id: int) -> dict | None:
        df = find_policy(self.db_path, int(policy_id))
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def find_claim(self, claim_id: str) -> dict | None:
        """Backward-compatible lookup for the earlier synthetic MVP tests."""
        try:
            numeric_id = int(str(claim_id).split("-")[-1])
        except ValueError:
            return None
        with sqlite3.connect(self.db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "claim_anomalies" not in tables:
                return None
            row = conn.execute(
                "SELECT * FROM claim_anomalies WHERE IDpol = ?", (numeric_id,)
            ).fetchone()
            if row is None:
                return None
            columns = [col[1] for col in conn.execute("PRAGMA table_info(claim_anomalies)").fetchall()]
        return dict(zip(columns, row))

    def top_anomalies(self, limit: int = 10) -> list[dict]:
        return top_anomalies(self.db_path, limit).to_dict(orient="records")

    def anomaly_region_summary(self) -> list[dict]:
        return anomaly_region_summary(self.db_path).to_dict(orient="records")

    def search_policy(self, query: str, k: int = 3) -> list[dict]:
        return self.retriever.search(query, k=k)

    def rag_status(self) -> dict:
        try:
            count = self.retriever.collection.count()
        except Exception:
            count = 0
        return {"indexed_documents": count, "ready": count > 0}

    def database_health(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            policies = int(conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0])
            anomalies = (
                int(conn.execute("SELECT COUNT(*) FROM claim_anomalies").fetchone()[0])
                if "claim_anomalies" in tables
                else 0
            )
        return {"policies": policies, "anomaly_rows": anomalies}
