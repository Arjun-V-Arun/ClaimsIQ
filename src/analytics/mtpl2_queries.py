"""SQL analytics for the real French MTPL2 insurance dataset."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

KPI_SQL = """
SELECT
    COUNT(*) AS policies,
    SUM(has_claim) AS policies_with_claims,
    SUM(ClaimNb) AS total_claims,
    ROUND(SUM(Exposure), 2) AS total_exposure_years,
    ROUND(SUM(total_claim_amount), 2) AS total_claim_amount,
    ROUND(SUM(ClaimNb) * 1.0 / NULLIF(SUM(Exposure), 0), 5) AS claims_per_exposure_year,
    ROUND(SUM(total_claim_amount) / NULLIF(SUM(ClaimNb), 0), 2) AS avg_claim_severity_eur,
    ROUND(SUM(total_claim_amount) / NULLIF(SUM(Exposure), 0), 2) AS loss_cost_per_exposure_year_eur
FROM policies;
"""

REGIONAL_SQL = """
WITH regional AS (
    SELECT
        Region AS region,
        COUNT(*) AS policy_count,
        SUM(has_claim) AS policies_with_claims,
        SUM(ClaimNb) AS claim_count,
        ROUND(SUM(Exposure), 2) AS exposure_years,
        ROUND(100.0 * SUM(has_claim) / COUNT(*), 2) AS pct_policies_with_claim,
        SUM(ClaimNb) * 1.0 / NULLIF(SUM(Exposure), 0) AS claims_per_exposure_year,
        ROUND(SUM(total_claim_amount), 2) AS total_claim_amount_eur,
        ROUND(SUM(total_claim_amount) / NULLIF(SUM(ClaimNb), 0), 2) AS avg_claim_severity_eur,
        ROUND(SUM(total_claim_amount) / NULLIF(SUM(Exposure), 0), 2) AS loss_cost_per_exposure_year_eur
    FROM policies
    GROUP BY Region
), ranked AS (
    SELECT
        region,
        policy_count,
        policies_with_claims,
        claim_count,
        exposure_years,
        pct_policies_with_claim,
        ROUND(claims_per_exposure_year, 5) AS claims_per_exposure_year,
        total_claim_amount_eur,
        avg_claim_severity_eur,
        loss_cost_per_exposure_year_eur,
        DENSE_RANK() OVER (ORDER BY claims_per_exposure_year DESC) AS frequency_rank
    FROM regional
)
SELECT *
FROM ranked
ORDER BY frequency_rank, region;
"""

REGIONAL_WINDOW_SQL = """
WITH regional AS (
    SELECT
        Region AS region,
        COUNT(*) AS policy_count,
        SUM(ClaimNb) AS claim_count,
        SUM(Exposure) AS exposure_years,
        SUM(ClaimNb) * 1.0 / NULLIF(SUM(Exposure), 0) AS claims_per_exposure_year
    FROM policies
    GROUP BY Region
), ranked AS (
    SELECT
        region,
        policy_count,
        claim_count,
        ROUND(exposure_years, 2) AS exposure_years,
        ROUND(claims_per_exposure_year, 5) AS claims_per_exposure_year,
        DENSE_RANK() OVER (ORDER BY claims_per_exposure_year DESC) AS frequency_rank
    FROM regional
)
SELECT * FROM ranked ORDER BY frequency_rank, region;
"""

FUEL_SQL = """
SELECT
    VehGas AS fuel_type,
    COUNT(*) AS policy_count,
    SUM(has_claim) AS policies_with_claims,
    SUM(ClaimNb) AS claim_count,
    ROUND(SUM(ClaimNb) * 1.0 / NULLIF(SUM(Exposure), 0), 5) AS claims_per_exposure_year,
    ROUND(SUM(total_claim_amount) / NULLIF(SUM(ClaimNb), 0), 2) AS avg_claim_severity_eur,
    ROUND(SUM(total_claim_amount), 2) AS total_claim_amount_eur
FROM policies
GROUP BY VehGas
ORDER BY claims_per_exposure_year DESC;
"""

DRIVER_AGE_SQL = """
WITH bands AS (
    SELECT
        CASE
            WHEN DrivAge < 25 THEN '<25'
            WHEN DrivAge < 35 THEN '25-34'
            WHEN DrivAge < 45 THEN '35-44'
            WHEN DrivAge < 55 THEN '45-54'
            WHEN DrivAge < 65 THEN '55-64'
            ELSE '65+'
        END AS driver_age_band,
        ClaimNb,
        Exposure,
        total_claim_amount
    FROM policies
)
SELECT
    driver_age_band,
    COUNT(*) AS policy_count,
    SUM(ClaimNb) AS claim_count,
    ROUND(SUM(ClaimNb) * 1.0 / NULLIF(SUM(Exposure), 0), 5) AS claims_per_exposure_year,
    ROUND(SUM(total_claim_amount) / NULLIF(SUM(ClaimNb), 0), 2) AS avg_claim_severity_eur
FROM bands
GROUP BY driver_age_band
ORDER BY
    CASE driver_age_band
        WHEN '<25' THEN 1
        WHEN '25-34' THEN 2
        WHEN '35-44' THEN 3
        WHEN '45-54' THEN 4
        WHEN '55-64' THEN 5
        ELSE 6
    END;
"""

ANOMALY_SQL = """
SELECT *
FROM claim_anomalies
ORDER BY anomaly_score DESC
LIMIT ?;
"""

POLICY_SQL = """
SELECT *
FROM claim_anomalies
WHERE IDpol = ?;
"""

ANOMALY_REGION_SQL = """
SELECT
    region,
    policy_count,
    flagged_count,
    flag_rate_pct,
    portfolio_share_pct,
    flag_share_pct,
    flag_rate_vs_portfolio_ratio
FROM anomaly_region_summary
ORDER BY flag_rate_vs_portfolio_ratio DESC, region;
"""



def run_query(db_path: Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def kpis(db_path: Path) -> pd.DataFrame:
    return run_query(db_path, KPI_SQL)


def regional_summary(db_path: Path) -> pd.DataFrame:
    return run_query(db_path, REGIONAL_SQL)


def regional_frequency_rank(db_path: Path) -> pd.DataFrame:
    return run_query(db_path, REGIONAL_WINDOW_SQL)


def fuel_summary(db_path: Path) -> pd.DataFrame:
    return run_query(db_path, FUEL_SQL)


def driver_age_summary(db_path: Path) -> pd.DataFrame:
    return run_query(db_path, DRIVER_AGE_SQL)


def find_policy(db_path: Path, policy_id: int) -> pd.DataFrame:
    return run_query(db_path, POLICY_SQL, (policy_id,))


def top_anomalies(db_path: Path, limit: int = 20) -> pd.DataFrame:
    return run_query(db_path, ANOMALY_SQL, (limit,))


def anomaly_region_summary(db_path: Path) -> pd.DataFrame:
    # The bootstrap materialises this table once. Fall back to the previous
    # on-demand aggregation for older databases so the API remains compatible.
    with sqlite3.connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='anomaly_region_summary'"
        ).fetchone()
    if exists:
        return run_query(db_path, ANOMALY_REGION_SQL)

    fallback_sql = """
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
    ORDER BY flag_rate_vs_portfolio_ratio DESC, region;
    """
    return run_query(db_path, fallback_sql)
