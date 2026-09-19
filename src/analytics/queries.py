"""Reusable SQL analytics for the claims database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


KPI_SQL = """
SELECT
    COUNT(*) AS total_claims,
    ROUND(AVG(claim_amount), 2) AS avg_claim_amount,
    ROUND(AVG(resolution_days), 2) AS avg_resolution_days,
    ROUND(100.0 * AVG(CASE WHEN claim_status = 'Rejected' THEN 1.0 ELSE 0 END), 2) AS rejection_rate_pct
FROM claims;
"""

REGIONAL_SQL = """
SELECT
    region,
    COUNT(*) AS claim_count,
    ROUND(SUM(claim_amount), 2) AS total_claim_amount,
    ROUND(AVG(claim_amount), 2) AS avg_claim_amount,
    ROUND(AVG(resolution_days), 2) AS avg_resolution_days,
    ROUND(100.0 * AVG(CASE WHEN claim_status = 'Rejected' THEN 1.0 ELSE 0 END), 2) AS rejection_rate_pct
FROM claims
GROUP BY region
ORDER BY total_claim_amount DESC;
"""

MONTHLY_SQL = """
WITH monthly AS (
    SELECT
        substr(claim_date, 1, 7) AS claim_month,
        COUNT(*) AS claim_count,
        SUM(claim_amount) AS total_claim_amount,
        AVG(resolution_days) AS avg_resolution_days
    FROM claims
    GROUP BY substr(claim_date, 1, 7)
)
SELECT
    claim_month,
    claim_count,
    ROUND(total_claim_amount, 2) AS total_claim_amount,
    ROUND(avg_resolution_days, 2) AS avg_resolution_days,
    ROUND(
        100.0 * (claim_count - LAG(claim_count) OVER (ORDER BY claim_month)) /
        NULLIF(LAG(claim_count) OVER (ORDER BY claim_month), 0), 2
    ) AS claim_count_mom_pct
FROM monthly
ORDER BY claim_month;
"""

CUSTOMER_WINDOW_SQL = """
SELECT
    customer_id,
    claim_id,
    claim_date,
    claim_amount,
    SUM(claim_amount) OVER (
        PARTITION BY customer_id
        ORDER BY claim_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3_claim_amount
FROM claims
ORDER BY customer_id, claim_date;
"""


def run_query(db_path: Path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def kpis(db_path: Path) -> pd.DataFrame:
    return run_query(db_path, KPI_SQL)


def regional_summary(db_path: Path) -> pd.DataFrame:
    return run_query(db_path, REGIONAL_SQL)


def monthly_trends(db_path: Path) -> pd.DataFrame:
    return run_query(db_path, MONTHLY_SQL)


def customer_rolling_claims(db_path: Path) -> pd.DataFrame:
    return run_query(db_path, CUSTOMER_WINDOW_SQL)
