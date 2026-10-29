# Applicant-Level Credit Delinquency Analysis

## Overview

This project studies applicant-level characteristics associated with observed serious credit delinquency using application data and monthly credit-history records.

The analysis covers data cleaning, dataset merging, feature engineering, exploratory data analysis (EDA), and statistical hypothesis testing.

## Research Questions

- Does observed serious-delinquency prevalence differ between selected education groups?
- Does income differ between applicants with and without observed serious delinquency?
- Is observed serious delinquency associated with income type?
- Do age distributions differ between applicants with and without observed serious delinquency?

## Methodology

1. Load and inspect the application and credit-history datasets.
2. Clean records and validate applicant IDs and monthly credit-history keys.
3. Merge the datasets and engineer age and employment features.
4. Define serious delinquency from observed monthly credit-status codes 2–5, considering eligible active-credit months.
5. Aggregate monthly records to the applicant level and split applicants into exploration and confirmation subsets.
6. Perform EDA and statistical hypothesis tests.
7. Audit the encoded dataset dimensions against the 3-million-cell requirement.

## Statistical Tests

- Two-proportion z-test for education groups.
- Welch's t-test for log-transformed income.
- Chi-square test of independence for income type.
- Mann–Whitney U test for age.

## Files

- `Dsc_project.ipynb`: Main notebook containing the analysis, outputs, and visualizations.
- `Dsc_project.py`: Python script exported from the notebook.
- `requirements.txt`: Python package dependencies.

## Data

The source datasets are not included in this repository. Place the required input files in the expected location and update the notebook's file paths if necessary.

## Reproducibility

Use the Python environment and package versions specified in `requirements.txt`. Run the notebook cells in order from top to bottom.

The findings describe associations with **observed serious delinquency** in the available credit history; they should not be interpreted as a validated model for predicting future default.