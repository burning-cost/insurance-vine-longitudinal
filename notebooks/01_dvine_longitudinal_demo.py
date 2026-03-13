# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-vine-longitudinal: D-vine copula for longitudinal insurance claims
# MAGIC
# MAGIC This notebook demonstrates the full workflow:
# MAGIC 1. Generate a synthetic insurance panel (multi-year policyholder data)
# MAGIC 2. Fit a `TwoPartDVine` model
# MAGIC 3. Predict next-year claim probability and severity given history
# MAGIC 4. Extract experience relativity tables
# MAGIC 5. Compare against NCD scale
# MAGIC
# MAGIC **Reference**: Yang & Czado (2022), Scandinavian Journal of Statistics

# COMMAND ----------

# MAGIC %pip install insurance-vine-longitudinal pyvinecopulib numpy scipy pandas scikit-learn statsmodels

# COMMAND ----------

import numpy as np
import pandas as pd
import warnings

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic panel data
# MAGIC
# MAGIC Mimics a UK motor insurer panel: 500 policyholders, 4 years each,
# MAGIC roughly 18% claim frequency, gamma-distributed severity.

# COMMAND ----------

def generate_insurance_panel(
    n_policies: int = 500,
    n_years: int = 4,
    base_claim_rate: float = 0.18,
    severity_mean: float = 2500.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic longitudinal insurance panel.

    Policyholders have persistent latent risk (lognormal). Each year's claim
    depends on both the latent risk and a temporal autocorrelation component.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_policies):
        # Policyholder latent risk
        risk = rng.lognormal(0, 0.4)
        p_base = np.clip(base_claim_rate * risk, 0.02, 0.80)

        # Covariate: age and vehicle group
        age = rng.integers(17, 75)
        veh_group = rng.integers(1, 6)

        prev_claim = 0
        for t in range(n_years):
            # Temporal dependence: history boosts future claim probability
            p_t = np.clip(p_base * (1.3 if prev_claim else 1.0), 0.01, 0.90)
            has_claim = int(rng.random() < p_t)
            amount = 0.0
            if has_claim:
                amount = float(rng.gamma(2.0, scale=severity_mean / 2.0))
            prev_claim = has_claim

            rows.append({
                "policy_id": f"POL{i:05d}",
                "year": 2020 + t,
                "has_claim": has_claim,
                "claim_amount": amount,
                "age": int(age),
                "vehicle_group": int(veh_group),
            })

    return pd.DataFrame(rows)


panel_df = generate_insurance_panel(n_policies=500, n_years=4, seed=42)
print(f"Panel shape: {panel_df.shape}")
print(f"Claim frequency: {panel_df['has_claim'].mean():.3f}")
print(f"Mean severity (when claimed): {panel_df.loc[panel_df['has_claim']==1, 'claim_amount'].mean():.0f}")
panel_df.head(12)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build PanelDataset

# COMMAND ----------

from insurance_vine_longitudinal import PanelDataset

panel = PanelDataset.from_dataframe(
    panel_df,
    id_col="policy_id",
    year_col="year",
    claim_col="has_claim",
    severity_col="claim_amount",
    covariate_cols=["age", "vehicle_group"],
)

print(f"Policies: {panel.n_policies}")
print(f"Max years: {panel.max_years}")
print()
print("Panel summary:")
print(panel.summary().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit TwoPartDVine

# COMMAND ----------

from insurance_vine_longitudinal import TwoPartDVine

model = TwoPartDVine(
    severity_family="gamma",
    max_truncation=2,  # BIC selects between p=1 and p=2
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(panel)

print(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inspect fitted vines

# COMMAND ----------

occ_vine = model.occurrence_vine
print("Occurrence vine:")
print(f"  T = {occ_vine.t_dim}")
print(f"  Selected truncation p = {occ_vine.truncation_level}")
print(f"  BIC by level: {occ_vine.fit_result_.bic_by_level}")
print(f"  Pair copula families: {occ_vine.fit_result_.family_counts}")

if model.severity_vine:
    sev_vine = model.severity_vine
    print("\nSeverity vine:")
    print(f"  T = {sev_vine.t_dim}")
    print(f"  Selected truncation p = {sev_vine.truncation_level}")
    print(f"  BIC by level: {sev_vine.fit_result_.bic_by_level}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Predict for held-out policyholders

# COMMAND ----------

# Use first 30 policyholders as test set
test_ids = panel.policy_ids[:30]
history_df = panel_df[panel_df["policy_id"].isin(test_ids)].copy()

# Next-year claim probability
proba = model.predict_proba(history_df)
print("Conditional claim probability (sample):")
print(proba.head(10).round(4))

# COMMAND ----------

# Severity quantiles
if model.severity_vine:
    q = model.predict_severity_quantile(history_df, quantiles=[0.5, 0.75, 0.95])
    print("\nConditional severity quantiles (sample):")
    print(q.head(10).round(0))

# COMMAND ----------

# Experience-rated premium
premium = model.predict_premium(history_df, loading=0.20)
print("\nExperience-rated premium with 20% loading (sample):")
print(premium.head(10).round(2))

# COMMAND ----------

# Experience relativity
relativity = model.experience_relativity(history_df)
print("\nExperience relativity (copula / a priori):")
print(relativity.describe().round(4))
print("\nSample:")
print(relativity.head(10).round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Relativity table
# MAGIC
# MAGIC The output pricing analysts use directly: given N years of history and
# MAGIC K claims, what is the experience relativity factor?

# COMMAND ----------

from insurance_vine_longitudinal import extract_relativity_curve

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    curve = extract_relativity_curve(
        model,
        claim_counts=[0, 1, 2, 3],
        n_years_list=[1, 2, 3, 4, 5],
        n_sim=100,
        seed=42,
    )

pivot = curve.pivot(index="claim_count", columns="n_years", values="relativity")
pivot.index.name = "claims"
pivot.columns.name = "years"
print("\nExperience relativity table:")
print(pivot.round(3).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. NCD comparison

# COMMAND ----------

from insurance_vine_longitudinal import compare_to_ncd

comparison = compare_to_ncd(curve)
print("Copula vs NCD (claim-free policyholders only):")
claim_free = comparison[comparison["claim_count"] == 0]
print(claim_free[["n_years", "vine_relativity", "ncd_relativity", "difference"]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Visualisation (if matplotlib available)

# COMMAND ----------

try:
    import matplotlib.pyplot as plt
    from insurance_vine_longitudinal._plot import (
        plot_tau_by_lag,
        plot_experience_surface,
        plot_bic_by_truncation,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Kendall's tau by lag
    plot_tau_by_lag(model, part="occurrence", ax=axes[0])

    # BIC curve
    plot_bic_by_truncation(model, part="occurrence", ax=axes[1])

    # Relativity surface
    plot_experience_surface(curve, ax=axes[2])

    plt.tight_layout()
    display(fig)
    plt.close()

except Exception as e:
    print(f"Plotting skipped: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Temporal dependence validation
# MAGIC
# MAGIC Does the model capture genuine claim persistence?

# COMMAND ----------

# Compare P(claim | last year claimed) vs P(claim | last year claim-free)
# Using model predictions on the training panel

# Split history into claimers vs non-claimers in year 3 (most recent)
year3 = panel_df[panel_df["year"] == 2022]
claimer_ids = year3[year3["has_claim"] == 1]["policy_id"].tolist()
noclaim_ids = year3[year3["has_claim"] == 0]["policy_id"].tolist()

history_claimers = panel_df[panel_df["policy_id"].isin(claimer_ids[:50])]
history_noclaims = panel_df[panel_df["policy_id"].isin(noclaim_ids[:50])]

prob_claimers = model.predict_proba(history_claimers)
prob_noclaims = model.predict_proba(history_noclaims)

print("Temporal dependence validation:")
print(f"  P(claim | last year: claimed)      = {prob_claimers.mean():.4f}")
print(f"  P(claim | last year: claim-free)   = {prob_noclaims.mean():.4f}")
print(f"  Ratio: {prob_claimers.mean() / prob_noclaims.mean():.2f}x")
print(f"\n  (Marginal claim rate: {panel_df['has_claim'].mean():.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The `insurance-vine-longitudinal` library:
# MAGIC - Captures temporal claim persistence via D-vine copulas on GLM residuals
# MAGIC - Selects Markov order by BIC (p=1 or p=2 for typical insurance panels)
# MAGIC - Outputs conditional claim probability, severity quantiles, and pure risk premium
# MAGIC - Generates the relativity table that pricing analysts need directly
# MAGIC - Provides an auditable methodology for FCA Consumer Duty documentation
# MAGIC
# MAGIC **Reference**: Yang & Czado (2022), Scandinavian Journal of Statistics 49(4)
