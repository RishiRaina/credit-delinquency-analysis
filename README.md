# Credit Delinquency Analysis

A data science project investigating applicant characteristics and their association with serious credit delinquency using application-level and monthly credit-history data.

## Project Overview

This project explores patterns in credit delinquency by combining applicant information with observed monthly credit repayment statuses. It uses data cleaning, feature engineering, exploratory data analysis, statistical hypothesis testing, and visualisation to investigate factors associated with serious delinquency.

The analysis focuses on understanding relationships in the available data rather than building a model to predict future loan defaults.

## Research Questions

The project investigates the following questions:

1. Is serious credit delinquency associated with applicants' education levels?
2. Do applicants with and without serious delinquency differ in their income distributions?
3. Is serious delinquency associated with applicants' income types?
4. Do applicants with and without serious delinquency differ in their age distributions?

## Dataset

The project uses two CSV files:

- **Application data:** Contains applicant-level demographic, financial, and application-related information.
- **Credit history data:** Contains monthly credit repayment status records associated with applicants.

Both CSV files are included in the project folder. 

### Outcome Definition

The binary outcome, `EVER_SERIOUS_DELINQUENCY`, is derived from observed monthly credit-history records. An eligible applicant is classified as having experienced serious delinquency if at least one observed eligible monthly record has a credit status of 2, 3, 4, or 5.

This outcome represents observed delinquency in the available credit history; it should not be interpreted as a prediction of future default.

## Methodology

The analysis follows these steps:

1. **Data loading and inspection:** Load both CSV files and examine their structure, data types, and quality.
2. **Data cleaning and preprocessing:** Handle missing values, clean relevant variables, and prepare the datasets for analysis.
3. **Feature engineering:** Derive the serious delinquency outcome from eligible monthly credit-history records and prepare applicant-level features.
4. **Data integration:** Combine application information with the derived credit-history outcome.
5. **Exploratory data analysis:** Examine delinquency rates and compare applicant characteristics using appropriate plots and summary statistics.
6. **Statistical hypothesis testing:** Evaluate the research questions using a separate confirmation dataset.
7. **Interpretation:** Discuss the findings, limitations, and implications of the observed associations.

The data is divided into exploration and confirmation subsets using a stratified split with random seed `2026`. This separates exploratory analysis from the confirmatory statistical tests.

## Statistical Tests

The project uses the following tests:

| Research question | Statistical test |
|---|---|
| Education level and delinquency | Two-proportion z-test |
| Income distribution differences | Welch's two-sample t-test on log-transformed income |
| Income type and delinquency | Chi-square test of independence |
| Age distribution differences | Mann–Whitney U test |

The results should be interpreted in light of the assumptions and limitations of each test, including potentially small expected cell counts in contingency tables.

## Visualisations

The notebook includes visualisations examining:

- Delinquency rates across education categories, with confidence intervals.
- Income distributions using a logarithmic transformation.
- Age distributions for applicants with and without observed serious delinquency.
- Delinquency rates across income types.

## Project Structure

```text
credit-delinquency-analysis/
├── Dsc_project.ipynb
├── Dsc_project.py
├── README.md
├── application_data.csv
└── credit_history.csv
```


- `Dsc_project.ipynb` — Jupyter Notebook containing the analysis, code, statistical tests, and visualisations.
- `Dsc_project.py` — Python version of the analysis.
- `README.md` — Project overview, methodology, and instructions.
- `application_data.csv` — Applicant-level dataset.
- `credit_history.csv` — Monthly credit-history dataset.

## Requirements

The analysis uses Python and common data science libraries, including:

- pandas
- NumPy
- Matplotlib
- SciPy
- scikit-learn
- statsmodels

Install any missing libraries in your Python environment before running the analysis.

## How to Run

1. Clone or download this repository.
2. Ensure both CSV datasets are present in the project folder.
3. Open `Dsc_project.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
4. Verify that the dataset paths in the notebook point to the correct CSV files.
5. Run the notebook cells in order.

Alternatively, run the Python script from the project directory:

```bash
python Dsc_project.py
```

Ensure that the script uses the correct paths to both CSV files and that the required libraries are installed.

## Limitations

- The analysis identifies statistical associations and does not establish causation.
- The outcome depends on the observed credit-history period and the availability of eligible monthly records.
- Statistical significance does not necessarily imply practical significance.
- The findings may not generalise to other applicant populations or time periods.
- The analysis is not a production-ready credit-scoring or future-default prediction system.
