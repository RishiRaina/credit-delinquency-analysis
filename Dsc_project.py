"""
Credit Delinquency Analysis
===========================

Standalone Python script for:
1. Loading and validating application and credit-history data.
2. Cleaning and linking the datasets.
3. Defining applicant-level serious delinquency.
4. Auditing dataset dimensions before and after one-hot encoding.
5. Splitting data into exploration and confirmation subsets.
6. Exploring applicant characteristics through visualisations.
7. Conducting four statistical hypothesis tests.

Input files:
    application_record.csv
    credit_record.csv

Run:
    python Dsc_project.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse, stats
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from IPython.display import display


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 2026
ALPHA = 0.05
CELL_THRESHOLD = 3_000_000

BASE_DIR = Path(__file__).resolve().parent

APPLICATION_FILE = BASE_DIR / "application_record.csv"
CREDIT_FILE = BASE_DIR / "credit_record.csv"

np.random.seed(SEED)

warnings.filterwarnings("default")


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    """Print a clearly formatted section heading."""
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def print_dataframe(df, title=None, max_rows=30):
    """Print a DataFrame in a readable console format."""
    if title:
        print(f"\n{title}")

    if df.empty:
        print("(No rows to display)")
    else:
        print(df.head(max_rows).to_string(index=False))

        if len(df) > max_rows:
            print(f"... {len(df) - max_rows:,} additional rows omitted.")


def require_columns(df, required_columns, dataset_name):
    """Raise an informative error if required columns are missing."""
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def iqr_outlier_summary(series):
    """Summarise potential outliers using the 1.5-IQR rule."""
    x = series.dropna()

    if x.empty:
        return pd.Series({
            "Lower fence": np.nan,
            "Upper fence": np.nan,
            "Potential outlier count": 0,
            "Potential outlier percent": np.nan,
        })

    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = (x < lower) | (x > upper)

    return pd.Series({
        "Lower fence": lower,
        "Upper fence": upper,
        "Potential outlier count": int(outliers.sum()),
        "Potential outlier percent": 100 * outliers.mean(),
    })


def cramers_v_from_data(data):
    """Calculate Cramer's V for income type and delinquency."""
    table = pd.crosstab(
        data["NAME_INCOME_TYPE"].fillna("Unknown"),
        data["EVER_SERIOUS_DELINQUENCY"],
    )

    if min(table.shape) < 2:
        return np.nan

    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)

    if denominator <= 0:
        return np.nan

    return np.sqrt(chi2 / denominator)


# ============================================================
# 3. LOAD DATASETS
# ============================================================

def load_datasets():
    """Load the two CSV files from the project directory."""
    for file_path in [APPLICATION_FILE, CREDIT_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {file_path}\n"
                "Place both CSV files in the same folder as Dsc_project.py."
            )

    app_raw = pd.read_csv(
        APPLICATION_FILE,
        dtype={"ID": "string"},
        low_memory=False,
    )

    credit_raw = pd.read_csv(
        CREDIT_FILE,
        dtype={
            "ID": "string",
            "MONTHS_BALANCE": "string",
            "STATUS": "string",
        },
        low_memory=False,
    )

    require_columns(
        app_raw,
        [
            "ID",
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "OCCUPATION_TYPE",
            "CNT_FAM_MEMBERS",
        ],
        "Application dataset",
    )

    require_columns(
        credit_raw,
        ["ID", "MONTHS_BALANCE", "STATUS"],
        "Credit-history dataset",
    )

    print_section("DATASET OVERVIEW")

    print("APPLICATION DATA")
    print("Shape:", app_raw.shape)
    print("Unique IDs:", app_raw["ID"].nunique())
    print("Columns:", app_raw.columns.tolist())
    print("Missing values:", app_raw.isna().sum().sum())

    print("\nCREDIT-HISTORY DATA")
    print("Shape:", credit_raw.shape)
    print("Unique IDs:", credit_raw["ID"].nunique())
    print("Columns:", credit_raw.columns.tolist())
    print("Missing values:", credit_raw.isna().sum().sum())

    return app_raw, credit_raw


# ============================================================
# 4. INITIAL DATA QUALITY INSPECTION
# ============================================================

def inspect_data_quality(app_raw, credit_raw):
    """Inspect missing values, duplicates, and repeated identifiers."""
    print_section("INITIAL DATA QUALITY INSPECTION")

    missing_summary = pd.DataFrame({
        "Missing count": app_raw.isna().sum(),
        "Missing percent": app_raw.isna().mean() * 100,
    }).sort_values("Missing percent", ascending=False)

    print_dataframe(
        missing_summary.reset_index(names="Column").round(2),
        "Missing values in application data",
    )

    print(
        "\nExact duplicate application rows:",
        app_raw.duplicated().sum(),
    )
    print(
        "Exact duplicate credit-history rows:",
        credit_raw.duplicated().sum(),
    )

    repeated_app_ids = app_raw["ID"].duplicated(keep=False)

    print(
        "Application rows with repeated IDs:",
        repeated_app_ids.sum(),
    )
    print(
        "Number of repeated application IDs:",
        app_raw.loc[repeated_app_ids, "ID"].nunique(),
    )
    print(
        "Repeated IDs in credit history:",
        credit_raw["ID"].duplicated().sum(),
    )


# ============================================================
# 5. CLEAN AND VALIDATE DATASETS
# ============================================================

def clean_datasets(app_raw, credit_raw):
    """Clean duplicate records and validate application and credit data."""
    print_section("CLEANING AND VALIDATION")

    app = app_raw.copy()
    credit = credit_raw.copy()

    app_duplicates_removed = int(app.duplicated().sum())
    credit_duplicates_removed = int(credit.duplicated().sum())

    app = app.drop_duplicates().copy()
    credit = credit.drop_duplicates().copy()

    print(
        "Exact duplicate application rows removed:",
        app_duplicates_removed,
    )
    print(
        "Exact duplicate credit-history rows removed:",
        credit_duplicates_removed,
    )

    print("\nMissing application IDs:", app["ID"].isna().sum())
    print("Missing credit-history IDs:", credit["ID"].isna().sum())

    app = app.dropna(subset=["ID"]).copy()
    credit = credit.dropna(subset=["ID"]).copy()

    # Remove ambiguous application IDs only when they also appear
    # in the credit-history dataset.
    conflict_mask = app.duplicated("ID", keep=False)
    conflict_ids = set(app.loc[conflict_mask, "ID"])
    credit_ids = set(credit["ID"].unique())
    conflicts_in_credit = conflict_ids & credit_ids

    print("\nConflicting application IDs:", len(conflict_ids))
    print(
        "Conflicting IDs also present in credit data:",
        len(conflicts_in_credit),
    )

    if conflicts_in_credit:
        print(
            "WARNING: Conflicting application IDs also occur in "
            "credit history. Investigate these records before proceeding."
        )

        conflict_records = (
            app[app["ID"].isin(conflicts_in_credit)]
            .sort_values("ID")
        )
        print_dataframe(conflict_records, "Conflicting application records")

        raise ValueError(
            "Conflicting application IDs overlap with credit history. "
            "Resolve these records before continuing."
        )

    # Ambiguous application IDs without credit-history overlap
    # are excluded from the application dataset.
    if conflict_ids:
        app = app[~app["ID"].isin(conflict_ids)].copy()

    if not app["ID"].is_unique:
        raise ValueError("Application IDs must be unique after cleaning.")

    credit["MONTHS_BALANCE"] = pd.to_numeric(
        credit["MONTHS_BALANCE"],
        errors="coerce",
    )

    credit["STATUS"] = credit["STATUS"].str.strip()

    invalid_month = credit["MONTHS_BALANCE"].isna()
    invalid_status = (
        credit["STATUS"].isna()
        | credit["STATUS"].eq("")
    )

    print("\nRows with invalid month:", invalid_month.sum())
    print("Rows with missing/blank status:", invalid_status.sum())

    credit = credit[
        (~invalid_month) & (~invalid_status)
    ].copy()

    # Month indices must be integers.
    credit = credit[
        credit["MONTHS_BALANCE"] % 1 == 0
    ].copy()

    credit["MONTHS_BALANCE"] = (
        credit["MONTHS_BALANCE"].astype(int)
    )

    valid_statuses = {"0", "1", "2", "3", "4", "5", "C", "X"}
    unexpected_statuses = set(credit["STATUS"].unique()) - valid_statuses

    print("Unexpected status values:", unexpected_statuses)

    if unexpected_statuses:
        raise ValueError(
            "Unexpected credit status values were found. "
            "Investigate them before continuing."
        )

    duplicate_month_mask = credit.duplicated(
        subset=["ID", "MONTHS_BALANCE"],
        keep=False,
    )

    print(
        "Rows in repeated ID-month groups:",
        duplicate_month_mask.sum(),
    )

    if duplicate_month_mask.any():
        print_dataframe(
            credit.loc[duplicate_month_mask]
            .sort_values(["ID", "MONTHS_BALANCE"]),
            "Repeated applicant-month records",
        )

        raise ValueError(
            "Repeated ID-month keys need investigation before merging."
        )

    print("\nApplication IDs unique:", app["ID"].is_unique)
    print(
        "Credit ID-month keys unique:",
        not credit.duplicated(["ID", "MONTHS_BALANCE"]).any(),
    )

    return app, credit


# ============================================================
# 6. DATA DICTIONARY
# ============================================================

def show_data_dictionary():
    """Print the data dictionary used in the analysis."""
    data_dictionary = pd.DataFrame([
        ["ID", "Identifier", "Applicant identifier", "Not a predictor"],
        ["CODE_GENDER", "Nominal", "Recorded gender category", "Category"],
        ["FLAG_OWN_CAR", "Nominal/binary", "Owns a car", "Y/N"],
        ["FLAG_OWN_REALTY", "Nominal/binary", "Owns property", "Y/N"],
        ["CNT_CHILDREN", "Discrete", "Number of children", "Count"],
        ["AMT_INCOME_TOTAL", "Continuous", "Recorded annual income",
         "Dataset currency"],
        ["NAME_INCOME_TYPE", "Nominal", "Income/employment category",
         "Category"],
        ["NAME_EDUCATION_TYPE", "Ordinal categorical", "Education category",
         "Category"],
        ["NAME_FAMILY_STATUS", "Nominal", "Family/marital status", "Category"],
        ["NAME_HOUSING_TYPE", "Nominal", "Housing arrangement", "Category"],
        ["DAYS_BIRTH", "Continuous-derived",
         "Days before application birth reference", "Days; negative values"],
        ["DAYS_EMPLOYED", "Continuous-derived",
         "Employment duration field", "Days; sentinel possible"],
        ["OCCUPATION_TYPE", "Nominal", "Occupation category", "Category"],
        ["CNT_FAM_MEMBERS", "Discrete", "Number of family members", "Count"],
        ["MONTHS_BALANCE", "Discrete", "Month relative to record reference",
         "Integer month index"],
        ["STATUS", "Categorical", "Monthly credit status", "0–5, C, X"],
    ], columns=[
        "Feature",
        "Type",
        "Real-world interpretation",
        "Unit/range",
    ])

    print_section("DATA DICTIONARY")
    print(data_dictionary.to_string(index=False))


# ============================================================
# 7. LINK APPLICATION AND CREDIT-HISTORY DATA
# ============================================================

def merge_datasets(app, credit):
    """Link monthly credit-history records to application records."""
    print_section("DATASET LINKAGE")

    app_ids = set(app["ID"].unique())
    credit_ids = set(credit["ID"].unique())
    matched_ids = app_ids & credit_ids

    print("Unique application IDs:", len(app_ids))
    print("Unique credit-history IDs:", len(credit_ids))
    print("Matched applicants:", len(matched_ids))
    print("Unmatched application IDs:", len(app_ids - credit_ids))
    print("Unmatched credit-history IDs:", len(credit_ids - app_ids))

    merged = credit.merge(
        app,
        on="ID",
        how="inner",
        validate="many_to_one",
    )

    print("\nMERGED MONTHLY DATA")
    print("Rows:", len(merged))
    print("Columns:", len(merged.columns))
    print("Unique applicants:", merged["ID"].nunique())
    print("Credit rows before merge:", len(credit))
    print("Matched credit rows:", len(merged))

    if len(merged) > len(credit):
        raise ValueError(
            "The merge produced more rows than the credit-history table. "
            "Check application ID uniqueness."
        )

    if len(credit) > 0:
        retained = 100 * len(merged) / len(credit)
        print(f"Monthly credit records retained: {retained:.2f}%")

    return merged


# ============================================================
# 8. FEATURE ENGINEERING AND APPLICANT-LEVEL OUTCOME
# ============================================================

def engineer_features(merged, app):
    """Create derived features and the applicant-level outcome."""
    print_section("FEATURE ENGINEERING")

    merged = merged.copy()

    merged["AGE_YEARS"] = -merged["DAYS_BIRTH"] / 365.25

    merged["EMPLOYMENT_YEARS"] = np.where(
        merged["DAYS_EMPLOYED"] < 0,
        -merged["DAYS_EMPLOYED"] / 365.25,
        np.nan,
    )

    print("\nMissing values in merged data:")
    missing_counts = (
        merged.isna().sum()
        .sort_values(ascending=False)
        .rename("Missing count")
        .to_frame()
    )
    print(missing_counts.head(20).to_string())

    print(
        "\nDAYS_EMPLOYED positive sentinel values:",
        (merged["DAYS_EMPLOYED"] == 365243).sum(),
    )
    print(
        "Nonpositive income values:",
        (merged["AMT_INCOME_TOTAL"] <= 0).sum(),
    )

    # Active statuses are 0–5. C and X are excluded from the
    # eligible-month denominator.
    merged["IS_ACTIVE_MONTH"] = merged["STATUS"].isin(
        ["0", "1", "2", "3", "4", "5"]
    )

    merged["IS_SERIOUS_DELINQUENCY"] = merged["STATUS"].isin(
        ["2", "3", "4", "5"]
    )

    merged["MONTHLY_SERIOUS_DELINQUENCY"] = np.where(
        merged["IS_ACTIVE_MONTH"],
        merged["IS_SERIOUS_DELINQUENCY"].astype(int),
        np.nan,
    )

    credit_summary = (
        merged.groupby("ID")
        .agg(
            total_recorded_months=("STATUS", "size"),
            active_credit_months=("IS_ACTIVE_MONTH", "sum"),
            serious_delinquent_months=("IS_SERIOUS_DELINQUENCY", "sum"),
            first_observed_month=("MONTHS_BALANCE", "min"),
            last_observed_month=("MONTHS_BALANCE", "max"),
        )
        .reset_index()
    )

    credit_summary["ELIGIBLE"] = (
        credit_summary["active_credit_months"] > 0
    )

    credit_summary["EVER_SERIOUS_DELINQUENCY"] = (
        credit_summary["serious_delinquent_months"] > 0
    ).astype(int)

    applicant_all = credit_summary.merge(
        app,
        on="ID",
        how="left",
        validate="one_to_one",
    )

    applicant_all = applicant_all[
        applicant_all["ELIGIBLE"]
    ].copy()

    applicant_all["EVER_SERIOUS_DELINQUENCY"] = (
        applicant_all["EVER_SERIOUS_DELINQUENCY"].astype(int)
    )

    print(
        "\nMatched applicants before eligibility filter:",
        credit_summary["ID"].nunique(),
    )
    print(
        "Eligible applicants for applicant-level inference:",
        len(applicant_all),
    )

    outcome_counts = (
        applicant_all["EVER_SERIOUS_DELINQUENCY"]
        .value_counts()
        .rename(index={
            0: "No observed serious delinquency",
            1: "Ever observed serious delinquency",
        })
        .rename("Applicants")
    )

    print("\nApplicant-level outcome counts:")
    print(outcome_counts.to_string())

    monthly_counts = (
        merged["MONTHLY_SERIOUS_DELINQUENCY"]
        .value_counts(dropna=False)
        .sort_index()
        .rename("Monthly observations")
    )

    print("\nMonthly outcome counts:")
    print(monthly_counts.to_string())

    return merged, applicant_all


# ============================================================
# 9. DATASET DIMENSIONS AND ONE-HOT ENCODING AUDIT
# ============================================================

def encode_for_audit(df, categorical_cols, dataset_name):
    """
    One-hot encode categorical columns and calculate the resulting
    dataset dimensions.

    This is a dimensionality audit, not a predictive modelling pipeline.
    """
    data = df.copy()

    excluded = {"ID", "EVER_SERIOUS_DELINQUENCY"}

    feature_cols = [
        col for col in data.columns
        if col not in excluded
    ]

    categorical_cols = [
        col for col in categorical_cols
        if col in feature_cols
    ]

    numeric_cols = [
        col for col in feature_cols
        if col not in categorical_cols
        and pd.api.types.is_numeric_dtype(data[col])
    ]

    unhandled_cols = [
        col for col in feature_cols
        if col not in categorical_cols
        and col not in numeric_cols
    ]

    if unhandled_cols:
        raise ValueError(
            f"{dataset_name}: unhandled feature columns: {unhandled_cols}"
        )

    X_cat = (
        data[categorical_cols]
        .astype("string")
        .fillna("Missing")
    )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )

    X_encoded_cat = encoder.fit_transform(X_cat)

    X_numeric = data[numeric_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    X_numeric_sparse = sparse.csr_matrix(
        X_numeric.fillna(0).to_numpy(dtype=float)
    )

    X_encoded = sparse.hstack(
        [X_numeric_sparse, X_encoded_cat],
        format="csr",
    )

    n_rows, n_features = X_encoded.shape
    n_cells = n_rows * n_features

    result = {
        "Dataset": dataset_name,
        "Rows": n_rows,
        "Original predictor columns": len(feature_cols),
        "Numeric features retained": len(numeric_cols),
        "Categorical features encoded": len(categorical_cols),
        "One-hot columns": X_encoded_cat.shape[1],
        "Total encoded features": n_features,
        "Total cells": n_cells,
        "Meets 3M requirement": n_cells >= CELL_THRESHOLD,
    }

    print_section(dataset_name)
    print(f"Rows: {n_rows:,}")
    print(f"Predictor columns before encoding: {len(feature_cols):,}")
    print(f"Numeric features retained: {len(numeric_cols):,}")
    print(f"Categorical features encoded: {len(categorical_cols):,}")
    print(f"One-hot encoded columns: {X_encoded_cat.shape[1]:,}")
    print(f"Total encoded features: {n_features:,}")
    print(f"Total cells: {n_cells:,}")
    print(
        "Meets 3-million-cell requirement:",
        n_cells >= CELL_THRESHOLD,
    )

    return X_encoded, result


def audit_dataset_dimensions(
    app_raw,
    app,
    credit_raw,
    credit,
    merged,
    applicant_all,
):
    """Report dataset sizes before and after encoding."""
    print_section("DATASET DIMENSIONS BEFORE ONE-HOT ENCODING")

    dataset_stages = [
        ("Raw application dataset", app_raw),
        ("Cleaned application dataset", app),
        ("Raw credit-history dataset", credit_raw),
        ("Cleaned credit-history dataset", credit),
        ("Linked/engineered monthly dataset", merged),
        ("Eligible applicant-level dataset", applicant_all),
    ]

    stage_results = []

    for name, df in dataset_stages:
        rows, columns = df.shape

        result = {
            "Dataset stage": name,
            "Rows": rows,
            "Columns": columns,
            "Unique applicants": df["ID"].nunique(),
            "Missing cells": int(df.isna().sum().sum()),
        }

        stage_results.append(result)

        print(f"\n{name}")
        print(f"Rows: {rows:,}")
        print(f"Columns: {columns:,}")
        print(f"Unique IDs: {df['ID'].nunique():,}")
        print(f"Missing cells: {df.isna().sum().sum():,}")

    stage_audit = pd.DataFrame(stage_results)

    print("\nDATASET STAGE SUMMARY")
    print(stage_audit.to_string(index=False))

    monthly_categorical = [
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "STATUS",
    ]

    applicant_categorical = [
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
    ]

    _, applicant_audit = encode_for_audit(
        df=applicant_all,
        categorical_cols=applicant_categorical,
        dataset_name="Eligible applicant-level dataset",
    )

    _, monthly_audit = encode_for_audit(
        df=merged,
        categorical_cols=monthly_categorical,
        dataset_name="Linked/engineered monthly dataset",
    )

    encoding_audit = pd.DataFrame([
        applicant_audit,
        monthly_audit,
    ])

    print_section("FINAL ENCODING AUDIT")
    print(encoding_audit.to_string(index=False))

    print_section("3-MILLION-CELL REQUIREMENT CHECK")

    for result in [applicant_audit, monthly_audit]:
        print(
            f"\n{result['Dataset']}: "
            f"{result['Rows']:,} x "
            f"{result['Total encoded features']:,} = "
            f"{result['Total cells']:,} cells"
        )

        print(
            "Requirement met:",
            result["Total cells"] >= CELL_THRESHOLD,
        )


# ============================================================
# 10. EXPLORATION / CONFIRMATION SPLIT
# ============================================================

def split_data(applicant_all):
    """Create stratified exploration and confirmation subsets."""
    print_section("EXPLORATION AND CONFIRMATION SPLIT")

    analysis_df = applicant_all.dropna(
        subset=["EVER_SERIOUS_DELINQUENCY"]
    ).copy()

    analysis_df["EVER_SERIOUS_DELINQUENCY"] = (
        analysis_df["EVER_SERIOUS_DELINQUENCY"].astype(int)
    )

    explore_df, confirm_df = train_test_split(
        analysis_df,
        test_size=0.20,
        random_state=SEED,
        stratify=analysis_df["EVER_SERIOUS_DELINQUENCY"],
    )

    explore_df = explore_df.copy()
    confirm_df = confirm_df.copy()

    explore_df["AGE_YEARS"] = (
        -pd.to_numeric(
            explore_df["DAYS_BIRTH"],
            errors="coerce",
        ) / 365.25
    )

    confirm_df["AGE_YEARS"] = (
        -pd.to_numeric(
            confirm_df["DAYS_BIRTH"],
            errors="coerce",
        ) / 365.25
    )

    print("Exploration set:", explore_df.shape)
    print("Confirmation set:", confirm_df.shape)

    print("\nExploration subset numeric summaries:")
    summary = explore_df[
        [
            "AMT_INCOME_TOTAL",
            "AGE_YEARS",
            "CNT_CHILDREN",
            "CNT_FAM_MEMBERS",
        ]
    ].describe(
        percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]
    ).T

    print(summary.to_string())

    outlier_summary = pd.DataFrame({
        col: iqr_outlier_summary(explore_df[col])
        for col in [
            "AMT_INCOME_TOTAL",
            "AGE_YEARS",
            "CNT_CHILDREN",
            "CNT_FAM_MEMBERS",
        ]
    }).T

    print("\nPotential outlier summary:")
    print(outlier_summary.round(2).to_string())

    return explore_df, confirm_df


# ============================================================
# 11. EXPLORATORY DATA ANALYSIS
# ============================================================

def perform_eda(explore_df):
    """Generate the four exploratory visualisations."""
    print_section("EXPLORATORY DATA ANALYSIS")

    eda = explore_df.copy()

    target = "EVER_SERIOUS_DELINQUENCY"

    eda["AGE_YEARS"] = -eda["DAYS_BIRTH"] / 365.25
    eda["LOG_INCOME"] = np.log1p(eda["AMT_INCOME_TOTAL"])
    eda[target] = eda[target].astype(int)

    # Figure 1: Delinquency rate by education.
    education_summary = (
        eda.groupby("NAME_EDUCATION_TYPE", observed=True)
        .agg(
            n=(target, "size"),
            events=(target, "sum"),
        )
        .reset_index()
    )

    education_summary["rate"] = (
        education_summary["events"] / education_summary["n"]
    )

    confidence_intervals = education_summary.apply(
        lambda row: proportion_confint(
            count=int(row["events"]),
            nobs=int(row["n"]),
            alpha=ALPHA,
            method="wilson",
        ),
        axis=1,
    )

    education_summary["ci_low"] = [
        interval[0] for interval in confidence_intervals
    ]
    education_summary["ci_high"] = [
        interval[1] for interval in confidence_intervals
    ]

    education_summary = education_summary.sort_values("rate")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    y = np.arange(len(education_summary))
    rates = education_summary["rate"].to_numpy() * 100
    lower = (
        education_summary["rate"] - education_summary["ci_low"]
    ).to_numpy() * 100
    upper = (
        education_summary["ci_high"] - education_summary["rate"]
    ).to_numpy() * 100

    ax.errorbar(
        rates,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(education_summary["NAME_EDUCATION_TYPE"])
    ax.set_xlabel("Applicants with observed serious delinquency (%)")
    ax.set_ylabel("Education category")
    ax.set_title("Observed Serious-Delinquency Rate by Education")
    ax.grid(axis="x", alpha=0.25)

    for i, row in enumerate(education_summary.itertuples(index=False)):
        ax.annotate(
            f"n={row.n:,}",
            (row.rate * 100, i),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    fig.tight_layout()
    plt.show()

    # Figure 2: Log-income distribution by outcome.
    income_groups = [
        eda.loc[eda[target] == 0, "LOG_INCOME"].dropna(),
        eda.loc[eda[target] == 1, "LOG_INCOME"].dropna(),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.boxplot(
        income_groups,
        labels=[
            f"No observed serious delinquency\n(n={len(income_groups[0]):,})",
            f"Observed serious delinquency\n(n={len(income_groups[1]):,})",
        ],
        showfliers=False,
    )

    ax.set_ylabel("log(1 + annual income)")
    ax.set_title("Log-Income by Observed Delinquency Group")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    plt.show()

    # Figure 3: Age distribution by outcome.
    age_groups = [
        eda.loc[eda[target] == 0, "AGE_YEARS"].dropna(),
        eda.loc[eda[target] == 1, "AGE_YEARS"].dropna(),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.boxplot(
        age_groups,
        labels=[
            f"No observed serious delinquency\n(n={len(age_groups[0]):,})",
            f"Observed serious delinquency\n(n={len(age_groups[1]):,})",
        ],
        showfliers=False,
    )

    ax.set_ylabel("Age (years)")
    ax.set_title("Age by Observed Delinquency Group")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    plt.show()

    # Figure 4: Delinquency rate by income type.
    income_type_summary = (
        eda.groupby("NAME_INCOME_TYPE", observed=True)
        .agg(
            n=(target, "size"),
            events=(target, "sum"),
        )
        .reset_index()
    )

    income_type_summary["rate"] = (
        income_type_summary["events"] / income_type_summary["n"]
    )

    income_type_summary = income_type_summary.sort_values("rate")

    fig, ax = plt.subplots(figsize=(8.5, 5))

    y = np.arange(len(income_type_summary))

    ax.barh(
        y,
        income_type_summary["rate"] * 100,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(income_type_summary["NAME_INCOME_TYPE"])
    ax.set_xlabel("Applicants with observed serious delinquency (%)")
    ax.set_ylabel("Income type")
    ax.set_title("Observed Serious-Delinquency Rate by Income Type")
    ax.grid(axis="x", alpha=0.25)

    for i, row in enumerate(income_type_summary.itertuples(index=False)):
        ax.annotate(
            f"{row.rate * 100:.2f}% (n={row.n:,})",
            (row.rate * 100, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    plt.show()

    print_dataframe(
        education_summary,
        "Education summary",
    )
    print_dataframe(
        income_type_summary,
        "Income-type summary",
    )


# ============================================================
# 12. TEST 1: TWO-PROPORTION Z-TEST FOR EDUCATION
# ============================================================

def test_education(confirm_df, test_results):
    """Compare serious-delinquency proportions across two education groups."""
    print_section("TEST 1: TWO-PROPORTION Z-TEST")

    higher = confirm_df[
        confirm_df["NAME_EDUCATION_TYPE"] == "Higher education"
    ]

    secondary = confirm_df[
        confirm_df["NAME_EDUCATION_TYPE"]
        == "Secondary / secondary special"
    ]

    n1 = len(higher)
    n2 = len(secondary)

    if n1 == 0 or n2 == 0:
        raise ValueError(
            "One of the education groups has no observations."
        )

    x1 = int(higher["EVER_SERIOUS_DELINQUENCY"].sum())
    x2 = int(secondary["EVER_SERIOUS_DELINQUENCY"].sum())

    p1 = x1 / n1
    p2 = x2 / n2

    # H0: p1 = p2; H1: p1 != p2.
    z_stat, p_value = proportions_ztest(
        count=[x1, x2],
        nobs=[n1, n2],
        alternative="two-sided",
    )

    # Unpooled 95% confidence interval for p1 - p2.
    risk_difference = p1 - p2

    standard_error = np.sqrt(
        p1 * (1 - p1) / n1
        + p2 * (1 - p2) / n2
    )

    ci_low = (
        risk_difference - stats.norm.ppf(0.975) * standard_error
    )
    ci_high = (
        risk_difference + stats.norm.ppf(0.975) * standard_error
    )

    print("Higher-education group:", n1, "applicants")
    print("Secondary-education group:", n2, "applicants")
    print(f"Observed proportions: {p1:.4f} vs {p2:.4f}")
    print(
        "Risk difference (higher - secondary): "
        f"{risk_difference:.4f}"
    )
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"z statistic: {z_stat:.4f}")
    print("Degrees of freedom: not applicable")
    print(f"p-value: {p_value:.6g}")
    print("Practical threshold: 0.03 absolute risk difference")

    test_results.append({
        "Test": "Two-proportion z-test",
        "Statistic": z_stat,
        "df": np.nan,
        "p_raw": p_value,
        "Estimate": risk_difference,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "Estimate_type": "Risk difference",
        "Threshold": 0.03,
    })


# ============================================================
# 13. TEST 2: WELCH'S T-TEST FOR LOG-INCOME
# ============================================================

def test_income(confirm_df, test_results):
    """Compare mean log-income between delinquency groups."""
    print_section("TEST 2: WELCH'S T-TEST FOR LOG-INCOME")

    income_df = confirm_df[
        (confirm_df["AMT_INCOME_TOTAL"] > 0)
        & confirm_df["AMT_INCOME_TOTAL"].notna()
    ].copy()

    income_df["LOG_INCOME"] = np.log(
        income_df["AMT_INCOME_TOTAL"]
    )

    income_yes = income_df.loc[
        income_df["EVER_SERIOUS_DELINQUENCY"] == 1,
        "LOG_INCOME",
    ].to_numpy()

    income_no = income_df.loc[
        income_df["EVER_SERIOUS_DELINQUENCY"] == 0,
        "LOG_INCOME",
    ].to_numpy()

    if len(income_yes) < 2 or len(income_no) < 2:
        raise ValueError(
            "Both income groups must have at least two valid observations."
        )

    # H0: equal mean log-income; H1: unequal means.
    t_stat, p_value = stats.ttest_ind(
        income_yes,
        income_no,
        equal_var=False,
        alternative="two-sided",
    )

    n_yes, n_no = len(income_yes), len(income_no)
    mean_yes, mean_no = income_yes.mean(), income_no.mean()
    var_yes, var_no = income_yes.var(ddof=1), income_no.var(ddof=1)

    mean_diff = mean_yes - mean_no

    standard_error = np.sqrt(
        var_yes / n_yes + var_no / n_no
    )

    df_welch = (
        (var_yes / n_yes + var_no / n_no) ** 2
        / (
            (var_yes / n_yes) ** 2 / (n_yes - 1)
            + (var_no / n_no) ** 2 / (n_no - 1)
        )
    )

    t_critical = stats.t.ppf(0.975, df_welch)

    ci_low = mean_diff - t_critical * standard_error
    ci_high = mean_diff + t_critical * standard_error

    relative_diff = np.exp(mean_diff) - 1
    relative_ci_low = np.exp(ci_low) - 1
    relative_ci_high = np.exp(ci_high) - 1

    print("Applicants with serious delinquency:", n_yes)
    print("Applicants without observed serious delinquency:", n_no)
    print(f"Mean log-income difference: {mean_diff:.4f}")
    print(f"95% CI on log scale: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"t statistic: {t_stat:.4f}")
    print(f"Welch degrees of freedom: {df_welch:.2f}")
    print(f"p-value: {p_value:.6g}")
    print(
        "Relative geometric-mean difference: "
        f"{100 * relative_diff:.2f}%"
    )
    print(
        "95% CI for relative difference: "
        f"[{100 * relative_ci_low:.2f}%, "
        f"{100 * relative_ci_high:.2f}%]"
    )
    print("Practical threshold: 5% relative difference")

    test_results.append({
        "Test": "Welch t-test on log-income",
        "Statistic": t_stat,
        "df": df_welch,
        "p_raw": p_value,
        "Estimate": mean_diff,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "Estimate_type": "Mean log-income difference",
        "Threshold": np.log(1.05),
    })


# ============================================================
# 14. TEST 3: CHI-SQUARE TEST FOR INCOME TYPE
# ============================================================

def test_income_type(confirm_df, test_results):
    """Test the association between income type and delinquency."""
    print_section("TEST 3: CHI-SQUARE TEST OF INDEPENDENCE")

    ct_df = confirm_df.copy()

    ct_df["NAME_INCOME_TYPE"] = (
        ct_df["NAME_INCOME_TYPE"]
        .fillna("Unknown")
        .astype(str)
    )

    contingency = pd.crosstab(
        ct_df["NAME_INCOME_TYPE"],
        ct_df["EVER_SERIOUS_DELINQUENCY"],
    )

    if min(contingency.shape) < 2:
        raise ValueError(
            "The contingency table must contain at least two rows "
            "and two columns."
        )

    chi2_stat, p_value, df_chi, expected = chi2_contingency(
        contingency,
        correction=False,
    )

    n = contingency.to_numpy().sum()
    min_dim = min(
        contingency.shape[0] - 1,
        contingency.shape[1] - 1,
    )

    cramers_v = np.sqrt(chi2_stat / (n * min_dim))

    print("Contingency table:")
    print(contingency.to_string())

    print(f"\nChi-square statistic: {chi2_stat:.4f}")
    print(f"Degrees of freedom: {df_chi}")
    print(f"p-value: {p_value:.6g}")
    print(f"Cramer's V: {cramers_v:.4f}")
    print("Practical benchmark: Cramer's V = 0.10")
    print("Minimum expected cell count:", expected.min())
    print("Expected cells below 5:", (expected < 5).sum())

    if (expected < 5).any():
        print(
            "WARNING: Some expected cell counts are below 5. "
            "Interpret the chi-square approximation cautiously."
        )

    # Bootstrap confidence interval for Cramer's V.
    rng = np.random.default_rng(SEED)
    bootstrap_repetitions = 500
    bootstrap_values = []

    for _ in range(bootstrap_repetitions):
        indices = rng.integers(
            0,
            len(ct_df),
            size=len(ct_df),
        )
        sample = ct_df.iloc[indices]

        value = cramers_v_from_data(sample)

        if np.isfinite(value):
            bootstrap_values.append(value)

    if bootstrap_values:
        ci_low, ci_high = np.quantile(
            bootstrap_values,
            [0.025, 0.975],
        )
    else:
        ci_low, ci_high = np.nan, np.nan

    print(f"Bootstrap 95% CI for Cramer's V: [{ci_low:.4f}, {ci_high:.4f}]")

    test_results.append({
        "Test": "Chi-square independence",
        "Statistic": chi2_stat,
        "df": df_chi,
        "p_raw": p_value,
        "Estimate": cramers_v,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "Estimate_type": "Cramer's V",
        "Threshold": 0.10,
    })


# ============================================================
# 15. TEST 4: MANN-WHITNEY U TEST FOR AGE
# ============================================================

def test_age(confirm_df, test_results):
    """Compare age distributions between delinquency groups."""
    print_section("TEST 4: MANN-WHITNEY U TEST FOR AGE")

    age_df = confirm_df.dropna(
        subset=["AGE_YEARS", "EVER_SERIOUS_DELINQUENCY"]
    ).copy()

    age_yes = age_df.loc[
        age_df["EVER_SERIOUS_DELINQUENCY"] == 1,
        "AGE_YEARS",
    ].to_numpy(dtype=float)

    age_no = age_df.loc[
        age_df["EVER_SERIOUS_DELINQUENCY"] == 0,
        "AGE_YEARS",
    ].to_numpy(dtype=float)

    if len(age_yes) == 0 or len(age_no) == 0:
        raise ValueError(
            "At least one delinquency group has no valid age observations."
        )

    print("Valid applicants:", len(age_df))
    print(
        "Applicants with observed serious delinquency:",
        len(age_yes),
    )
    print(
        "Applicants without observed serious delinquency:",
        len(age_no),
    )

    u_stat, p_value = mannwhitneyu(
        age_yes,
        age_no,
        alternative="two-sided",
        method="asymptotic",
    )

    median_yes = np.median(age_yes)
    median_no = np.median(age_no)
    median_diff = median_yes - median_no

    # Bootstrap confidence interval for the difference in medians.
    rng = np.random.default_rng(SEED)
    bootstrap_repetitions = 1000
    bootstrap_differences = np.empty(bootstrap_repetitions)

    for i in range(bootstrap_repetitions):
        sample_yes = rng.choice(
            age_yes,
            size=len(age_yes),
            replace=True,
        )
        sample_no = rng.choice(
            age_no,
            size=len(age_no),
            replace=True,
        )

        bootstrap_differences[i] = (
            np.median(sample_yes) - np.median(sample_no)
        )

    ci_low, ci_high = np.quantile(
        bootstrap_differences,
        [0.025, 0.975],
    )

    print(f"Median age, serious delinquency: {median_yes:.3f} years")
    print(f"Median age, no serious delinquency: {median_no:.3f} years")
    print(f"Median age difference: {median_diff:.3f} years")
    print(f"Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"U statistic: {u_stat:.4f}")
    print("Degrees of freedom: not applicable")
    print(f"p-value: {p_value:.6g}")
    print("Practical threshold: 2 years in median age difference")

    test_results.append({
        "Test": "Mann-Whitney U",
        "Statistic": u_stat,
        "df": np.nan,
        "p_raw": p_value,
        "Estimate": median_diff,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "Estimate_type": "Median age difference (years)",
        "Threshold": 2.0,
    })


# ============================================================
# 16. RUN THE COMPLETE ANALYSIS
# ============================================================

def main():
    """Execute the complete analysis from raw data to hypothesis tests."""
    print_section("CREDIT DELINQUENCY ANALYSIS")
    print("Random seed:", SEED)
    print("Significance level:", ALPHA)
    print("Project directory:", BASE_DIR)

    # Load and inspect the input datasets.
    app_raw, credit_raw = load_datasets()

    inspect_data_quality(app_raw, credit_raw)

    # Clean and validate the data.
    app, credit = clean_datasets(app_raw, credit_raw)

    show_data_dictionary()

    # Link the two datasets and derive the analysis outcome.
    merged = merge_datasets(app, credit)

    merged, applicant_all = engineer_features(merged, app)

    # Audit original and one-hot-encoded dataset dimensions.
    audit_dataset_dimensions(
        app_raw=app_raw,
        app=app,
        credit_raw=credit_raw,
        credit=credit,
        merged=merged,
        applicant_all=applicant_all,
    )

    # Create exploration and confirmation subsets.
    explore_df, confirm_df = split_data(applicant_all)

    # Visualisations use the exploration subset.
    perform_eda(explore_df)

    # Statistical tests use the confirmation subset.
    test_results = []

    test_education(confirm_df, test_results)
    test_income(confirm_df, test_results)
    test_income_type(confirm_df, test_results)
    test_age(confirm_df, test_results)

    # Summarise the four statistical tests.
    results_df = pd.DataFrame(test_results)

    print_section("SUMMARY OF HYPOTHESIS TESTS")
    print(results_df.to_string(index=False))

    print_section("ANALYSIS COMPLETE")
    print("All four statistical tests have been executed.")
    print("No datasets or figures were saved to disk.")


if __name__ == "__main__":
    main()