# Breusch-Pagan Homoscedasticity Check

This Python script fits ordinary least squares (OLS) regression models and
applies the Breusch-Pagan test to each model's residuals. The test assesses
whether the residual variance is constant (homoscedasticity), an assumption
often checked in regression analysis.

The script checks these models when their required columns are present:

| Model | Predictors | Outcome |
| --- | --- | --- |
| 1 | `TTF` | `Productivity` |
| 2 | `TTF` | `Code_Quality` |
| 3a | `Job_Demands` | `Well_Being` |
| 3b | `Job_Resources` | `Well_Being` |
| 3c | `Job_Demands`, `Job_Resources` | `Well_Being` |

## Requirements

- Python 3
- pandas
- statsmodels

Install the required packages:

```bash
python -m pip install pandas statsmodels
```

## How to run

Pass the path to your CSV file as the script's only argument:

```bash
python breusch_pagan_check.py "path/to/your_data.csv"
```

For example, in Windows PowerShell:

```powershell
python .\breusch_pagan_check.py "C:\path\to\your_data.csv"
```

The CSV file should contain the predictor and outcome columns listed above.
Models whose required columns are missing are skipped. The script reads the
CSV file from the path you provide; no dataset is included in this repository.

## Interpreting the output

For each model, the script prints the Breusch-Pagan LM statistic and p-value.
Using a significance level of 0.05:

- **p-value > 0.05:** fail to reject the null hypothesis; the test does not
  detect evidence of heteroscedasticity.
- **p-value <= 0.05:** reject the null hypothesis; the test detects evidence
  of heteroscedasticity.

A p-value above 0.05 does not prove that the residual variance is constant.
