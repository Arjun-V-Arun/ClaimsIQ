"""Train an unsupervised anomaly detector over claims."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = ["claim_amount", "repair_cost", "device_age_months", "resolution_days", "fraud_signal"]
CATEGORICAL = ["device_type", "damage_type", "region", "channel"]


def detect_anomalies(db_path: Path, contamination: float = 0.025) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM claims", conn)

    features = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("features", features),
            (
                "model",
                IsolationForest(
                    n_estimators=250,
                    contamination=contamination,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(df[NUMERIC + CATEGORICAL])
    df["anomaly_score"] = -pipeline.decision_function(df[NUMERIC + CATEGORICAL])
    df["is_anomaly"] = pipeline.predict(df[NUMERIC + CATEGORICAL]) == -1

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS claim_anomalies")
        df.to_sql("claim_anomalies", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_anomalies_score ON claim_anomalies(anomaly_score)"
        )
        conn.commit()

    return df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--contamination", type=float, default=0.025)
    args = parser.parse_args()

    result = detect_anomalies(args.db, args.contamination)
    print(f"Flagged {result['is_anomaly'].sum():,} anomalous claims")
    print(result[result["is_anomaly"]].head(10)[[
        "claim_id", "claim_amount", "damage_type", "region", "anomaly_score"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
