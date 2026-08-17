"""
preprocessing_pipeline.py
==========================
Customer Personality Segmentation — Module 5 (CPS-M05)
Task 7: Reusable Preprocessing Pipeline

This script packages the full feature engineering + preprocessing workflow
(missing values -> feature creation -> encoding -> transformation -> scaling
-> final feature selection) into one function so it can be re-run on any
future export of the same customer dataset without redoing the analysis
by hand.

Usage
-----
As a script:
    python preprocessing_pipeline.py input.csv output.csv

As a module:
    from preprocessing_pipeline import run_preprocessing_pipeline
    df_ready = run_preprocessing_pipeline("marketing_campaign.csv", "output.csv")

Author: (Intern) — Customer Personality Segmentation Project
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# Config — kept as constants at the top so future interns can tweak them
# without digging through the function bodies.
# ---------------------------------------------------------------------------
SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
              "MntFishProducts", "MntSweetProducts", "MntGoldProds"]

PURCHASE_COLS = ["NumDealsPurchases", "NumWebPurchases",
                  "NumCatalogPurchases", "NumStorePurchases"]

CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                  "AcceptedCmp4", "AcceptedCmp5", "Response"]

CHANNEL_MAP = {
    "NumWebPurchases": "Web",
    "NumCatalogPurchases": "Catalog",
    "NumStorePurchases": "Store",
    "NumDealsPurchases": "Deals",
}

PRODUCT_MAP = {
    "MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
    "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold",
}

EDUCATION_ORDER = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
ACTIVITY_ORDER = {"Low": 0, "Medium": 1, "High": 2}

NOMINAL_COLS = ["Marital_Status", "Preferred_Shopping_Channel", "Product_Preference"]

NUMERIC_COLS_FOR_SKEW = [
    "Income", "Customer_Age", "Customer_Tenure_Days", "Recency",
    "Total_Spending", "Total_Purchases", "Avg_Spending_Per_Purchase",
    "Digital_Engagement", "Deal_Dependency",
] + SPEND_COLS

SKEW_THRESHOLD = 0.75

DROP_COLS = [
    "ID", "Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue",
    "Education", "Customer_Activity_Level", "Kidhome", "Teenhome",
]


# ---------------------------------------------------------------------------
# Step 1: Missing value handling
# ---------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing Income with the median and drop rows with an
    implausible Year_Birth (would make the customer 100+ years old)."""
    df = df.copy()
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])

    reference_date = df["Dt_Customer"].max() + pd.Timedelta(days=1)
    reference_year = reference_date.year

    before = len(df)
    df = df[df["Year_Birth"] >= reference_year - 100].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[handle_missing_values] dropped {dropped} rows with implausible Year_Birth")

    income_median = df["Income"].median()
    df["Income"] = df["Income"].fillna(income_median)

    df["Marital_Status"] = df["Marital_Status"].replace(
        {"Alone": "Single", "Absurd": "Single", "YOLO": "Single"}
    )

    df = df.drop_duplicates().reset_index(drop=True)

    # stash reference date/year as attrs so later steps can reuse them
    df.attrs["reference_date"] = reference_date
    df.attrs["reference_year"] = reference_year
    df.attrs["income_median"] = income_median
    return df


# ---------------------------------------------------------------------------
# Step 2: Feature creation
# ---------------------------------------------------------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all 13 behavioral features described in the Feature Dictionary."""
    df = df.copy()
    reference_date = df.attrs.get("reference_date", df["Dt_Customer"].max() + pd.Timedelta(days=1))
    reference_year = df.attrs.get("reference_year", reference_date.year)

    df["Customer_Age"] = reference_year - df["Year_Birth"]
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    partnered = df["Marital_Status"].isin(["Married", "Together"])
    df["Family_Size"] = 1 + partnered.astype(int) + df["Total_Children"]

    df["Total_Spending"] = df[SPEND_COLS].sum(axis=1)
    df["Total_Purchases"] = df[PURCHASE_COLS].sum(axis=1)
    df["Total_Campaign_Acceptance"] = df[CAMPAIGN_COLS].sum(axis=1)

    df["Avg_Spending_Per_Purchase"] = (
        df["Total_Spending"] / df["Total_Purchases"].replace(0, np.nan)
    ).fillna(0)

    df["Digital_Engagement"] = df["NumWebPurchases"] + df["NumWebVisitsMonth"]

    df["Deal_Dependency"] = (
        df["NumDealsPurchases"] / df["Total_Purchases"].replace(0, np.nan)
    ).fillna(0)

    channel_cols = list(CHANNEL_MAP.keys())

    def _preferred_channel(row):
        if row[channel_cols].sum() == 0:
            return "None"
        return CHANNEL_MAP[row[channel_cols].idxmax()]

    df["Preferred_Shopping_Channel"] = df[channel_cols].apply(_preferred_channel, axis=1)

    def _preferred_product(row):
        if row[SPEND_COLS].sum() == 0:
            return "None"
        return PRODUCT_MAP[row[SPEND_COLS].idxmax()]

    df["Product_Preference"] = df[SPEND_COLS].apply(_preferred_product, axis=1)

    df["Customer_Activity_Level"] = pd.cut(
        df["Recency"], bins=[-1, 30, 60, 100], labels=["High", "Medium", "Low"]
    )

    return df


# ---------------------------------------------------------------------------
# Step 3: Encoding
# ---------------------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode ordinal columns, one-hot encode nominal columns."""
    df = df.copy()
    df["Education_Encoded"] = df["Education"].map(EDUCATION_ORDER)
    df["Customer_Activity_Level_Encoded"] = df["Customer_Activity_Level"].map(ACTIVITY_ORDER)
    df = pd.get_dummies(df, columns=NOMINAL_COLS, prefix=NOMINAL_COLS, dtype=int)
    return df


# ---------------------------------------------------------------------------
# Step 4: Feature transformation (skew correction)
# ---------------------------------------------------------------------------
def transform_features(df: pd.DataFrame, threshold: float = SKEW_THRESHOLD) -> pd.DataFrame:
    """Apply log1p to any numeric column whose |skew| exceeds the threshold."""
    df = df.copy()
    for col in NUMERIC_COLS_FOR_SKEW:
        if col not in df.columns:
            continue
        sk = df[col].skew()
        if abs(sk) > threshold:
            df[col + "_log"] = np.log1p(df[col].clip(lower=0))
    return df


# ---------------------------------------------------------------------------
# Step 5: Scaling
# ---------------------------------------------------------------------------
def scale_features(df: pd.DataFrame):
    """Fit a RobustScaler on the numeric feature set and append *_scaled columns.
    Returns (df_with_scaled_cols, fitted_scaler, list_of_scaled_source_columns)."""
    df = df.copy()
    log_cols = [c for c in df.columns if c.endswith("_log")]
    base_numeric = [c for c in NUMERIC_COLS_FOR_SKEW if (c + "_log") not in df.columns and c in df.columns]
    scale_cols = base_numeric + log_cols + [
        "Total_Children", "Family_Size", "Total_Campaign_Acceptance",
        "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
        "NumStorePurchases", "NumWebVisitsMonth", "Education_Encoded",
        "Customer_Activity_Level_Encoded",
    ]
    scale_cols = [c for c in dict.fromkeys(scale_cols) if c in df.columns]

    scaler = RobustScaler()
    scaled_values = scaler.fit_transform(df[scale_cols])
    scaled_col_names = [c + "_scaled" for c in scale_cols]
    df[scaled_col_names] = scaled_values

    return df, scaler, scale_cols


# ---------------------------------------------------------------------------
# Step 6: Final feature selection
# ---------------------------------------------------------------------------
def select_final_features(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    """Keep only the ID plus the scaled numeric columns and one-hot dummy
    columns — the feature set that's actually ready for clustering."""
    scaled_cols = [c for c in df.columns if c.endswith("_scaled")]
    dummy_cols = [c for c in df.columns if c.startswith((
        "Marital_Status_", "Preferred_Shopping_Channel_", "Product_Preference_"
    ))]
    keep_cols = ([id_col] if id_col in df.columns else []) + scaled_cols + dummy_cols
    return df[keep_cols].copy()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_preprocessing_pipeline(input_csv: str, output_csv: str = None,
                                also_save_full: str = None) -> pd.DataFrame:
    """Run the complete pipeline end to end.

    Parameters
    ----------
    input_csv : path to a raw customer CSV with the same schema as
        marketing_campaign.csv (ID, Year_Birth, Education, Marital_Status,
        Income, Kidhome, Teenhome, Dt_Customer, Recency, Mnt*, Num*Purchases,
        AcceptedCmp1-5, Complain, Z_CostContact, Z_Revenue, Response).
    output_csv : optional path to write the final ML-ready dataset to.
    also_save_full : optional path to also save the full engineered
        (pre-selection) dataset, useful for EDA / debugging.

    Returns
    -------
    pd.DataFrame — the final, ML-ready, fully numeric dataset.
    """
    raw = pd.read_csv(input_csv)

    df = handle_missing_values(raw)
    df = create_features(df)
    df = encode_features(df)
    df = transform_features(df)
    df, scaler, scale_cols = scale_features(df)

    if also_save_full:
        df.to_csv(also_save_full, index=False)

    df_ready = select_final_features(df)

    if output_csv:
        df_ready.to_csv(output_csv, index=False)
        print(f"[run_preprocessing_pipeline] wrote {len(df_ready)} rows, "
              f"{df_ready.shape[1]} columns -> {output_csv}")

    return df_ready


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocessing_pipeline.py <input_csv> [output_csv]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "ml_ready_output.csv"
    run_preprocessing_pipeline(in_path, out_path)
