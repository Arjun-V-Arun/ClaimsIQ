"""Unsupervised statistical review signals for real MTPL2 records.

This is a prioritisation aid, not a fraud classifier. The feature set avoids
using Region/Area as model inputs so geography does not become the dominant
source of flags merely because of local feature distributions.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

NUMERIC = [
    "ClaimNb",
    "Exposure",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "Density",
    "total_claim_amount",
    "claim_severity_per_claim",
]
CATEGORICAL = ["VehBrand", "VehGas"]
LOG_COLUMNS = ["Density", "total_claim_amount", "claim_severity_per_claim"]


def materialize_anomaly_tables(db_path: Path, df: pd.DataFrame) -> None:
    """Persist anomaly rows, indexes and a pre-aggregated region diagnostic."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS claim_anomalies")
        df.to_sql("claim_anomalies", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claim_anomaly_score "
            "ON claim_anomalies(anomaly_score DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claim_anomaly_region "
            "ON claim_anomalies(Region)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claim_anomaly_flag_region "
            "ON claim_anomalies(is_anomaly, Region)"
        )

        # Materialise the regional diagnostic so the conversational tool does
        # not repeatedly scan and aggregate the full anomaly table.
        conn.execute("DROP TABLE IF EXISTS anomaly_region_summary")
        conn.execute(
            """
            CREATE TABLE anomaly_region_summary AS
            WITH portfolio AS (
                SELECT Region, COUNT(*) AS policy_count
                FROM claim_anomalies
                GROUP BY Region
            ), flagged AS (
                SELECT Region, SUM(is_anomaly) AS flagged_count
                FROM claim_anomalies
                GROUP BY Region
            ), totals AS (
                SELECT COUNT(*) AS total_policies, SUM(is_anomaly) AS total_flagged
                FROM claim_anomalies
            )
            SELECT
                p.Region AS region,
                p.policy_count,
                COALESCE(f.flagged_count, 0) AS flagged_count,
                ROUND(100.0 * COALESCE(f.flagged_count, 0) / NULLIF(p.policy_count, 0), 3) AS flag_rate_pct,
                ROUND(100.0 * p.policy_count / NULLIF(t.total_policies, 0), 3) AS portfolio_share_pct,
                ROUND(100.0 * COALESCE(f.flagged_count, 0) / NULLIF(t.total_flagged, 0), 3) AS flag_share_pct,
                ROUND(
                    (COALESCE(f.flagged_count, 0) * 1.0 / NULLIF(p.policy_count, 0))
                    / NULLIF(t.total_flagged * 1.0 / NULLIF(t.total_policies, 0), 0),
                    3
                ) AS flag_rate_vs_portfolio_ratio
            FROM portfolio p
            LEFT JOIN flagged f ON f.Region = p.Region
            CROSS JOIN totals t
            ORDER BY flag_rate_vs_portfolio_ratio DESC, region
            """
        )
        conn.commit()


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df[NUMERIC + CATEGORICAL].copy()
    for col in LOG_COLUMNS:
        x[col] = np.log1p(x[col].clip(lower=0))
    return x


def detect_anomalies(db_path: Path, contamination: float = 0.02) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM policies", conn)

    if df.empty:
        raise ValueError("The policies table is empty")

    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("features", preprocessor),
            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    contamination=contamination,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    features = _feature_frame(df)
    pipeline.fit(features)

    raw_score = -pipeline.decision_function(features)
    df["raw_anomaly_score"] = raw_score
    # Percentile is an empirical extremeness measure within this dataset. It is
    # deliberately not presented as a probability of fraud or anomaly.
    df["anomaly_score"] = pd.Series(raw_score).rank(method="average", pct=True).to_numpy()
    df["anomaly_percentile_pct"] = df["anomaly_score"] * 100.0
    df["is_anomaly"] = pipeline.predict(features) == -1
    df["review_band"] = pd.cut(
        df["anomaly_score"],
        bins=[-np.inf, 0.90, 0.99, np.inf],
        labels=["Normal", "Watch", "High"],
    ).astype(str)

    materialize_anomaly_tables(db_path, df)
    return df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--contamination", type=float, default=0.02)
    args = parser.parse_args()
    result = detect_anomalies(args.db, args.contamination)
    print(f"Flagged {int(result['is_anomaly'].sum()):,} statistical review signals")
