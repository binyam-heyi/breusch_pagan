"""
======================================================================
Breusch-Pagan Homoscedasticity Test for Regression Models
======================================================================

Author: Binyam Heyi

Purpose:
--------
This script reads survey/regression data from a CSV file and tests
the regression models' residuals for homoscedasticity using the
Breusch-Pagan test.

The Breusch-Pagan test is a statistical method used to detect
heteroscedasticity in regression residuals.

Hypotheses:
-----------
H0: Homoscedasticity exists (constant residual variance).

H1: Heteroscedasticity exists (non-constant residual variance).

Interpretation:
---------------
p-value > 0.05:
    Fail to reject H0. The assumption of homoscedasticity is satisfied.

p-value <= 0.05:
    Reject H0. Evidence of heteroscedasticity is detected.

How to Run:
-----------
1. Open Command Prompt or PowerShell.

2. Navigate to the folder containing this Python script.

3. Run the script by providing the complete path to the CSV file:

       python breusch_pagan_check.py "C:\\path\\to\\your_data.csv"

4. Required packages can be installed using:

       pip install pandas statsmodels

Note:
-----
A CSV file path must be supplied when running the script.
No default data file path is specified in this script.

======================================================================
"""

import sys
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


def check_model(df, predictors, outcome, label):
    """
    Fit an OLS regression model and test residuals for
    homoscedasticity using the Breusch-Pagan test.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame containing the variables.

    predictors : list
        List of predictor variable names.

    outcome : str
        Name of the outcome/dependent variable.

    label : str
        Descriptive label for the regression model.
    """

    # Add a constant/intercept to the predictor variables.
    X = sm.add_constant(df[predictors])

    # Define the dependent variable.
    y = df[outcome]

    # Fit the ordinary least squares (OLS) regression model.
    model = sm.OLS(y, X).fit()

    # Obtain the regression residuals.
    residuals = model.resid

    # Perform the Breusch-Pagan test.
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)

    # Display results.
    print(f"\n{label}")
    print("-" * 70)

    print(f"Predictors: {', '.join(predictors)}")
    print(f"Outcome: {outcome}")

    print(
        f"Breusch-Pagan LM Test Statistic: "
        f"{bp_stat:.4f}"
    )

    print(f"p-value: {bp_p:.4f}")

    # Determine the result using the .05 significance level.
    verdict = (
        "HOMOSCEDASTICITY SATISFIED"
        if bp_p > 0.05
        else "HETEROSCEDASTICITY DETECTED"
    )

    print(f"Result: {verdict}")

    # Provide the statistical conclusion.
    if bp_p > 0.05:
        print(
            f"Conclusion: The regression assumption of "
            f"homoscedasticity is met "
            f"(p = {bp_p:.4f} > 0.05)."
        )
    else:
        print(
            f"Conclusion: Evidence of heteroscedasticity "
            f"(p = {bp_p:.4f} <= 0.05)."
        )


def main():
    r"""
    Main function.

    CSV data file can be provided as:
    1. Command-line argument: python breusch_pagan_check.py "path/to/file.csv"
    2. Environment variables:
       - DSRT930_DATA_PATH: Path to the data directory
       - DSRT930_CSV_FILE: Name of the CSV file
    3. The CSV path must be provided as a command-line argument.
    """

    # ---------------------------------------------------------------
    # CHECK FOR CSV FILE PATH
    # ---------------------------------------------------------------

    if len(sys.argv) != 2:
        print("Error: No CSV file path was provided.")
        print()
        print("Usage:")
        print(
            r'  python "path/to/breusch_pagan_check.py" '
            r'"path/to/your_data.csv"'
        )
        sys.exit(1)

    csv_path = sys.argv[1]

    # ---------------------------------------------------------------
    # READ DATA
    # ---------------------------------------------------------------

    try:
        df = pd.read_csv(csv_path)

    except FileNotFoundError:
        print(
            f"Error: CSV file not found at:\n{csv_path}"
        )
        print()
        print("Please verify that the file path is correct.")
        sys.exit(1)

    except Exception as e:
        print(
            f"Error reading CSV file:\n{e}"
        )
        sys.exit(1)

    # ---------------------------------------------------------------
    # DISPLAY ANALYSIS INFORMATION
    # ---------------------------------------------------------------

    print("=" * 70)

    print(
        "APPENDIX: BREUSCH-PAGAN HOMOSCEDASTICITY TEST RESULTS"
    )

    print("=" * 70)

    print(f"\nDataset: {csv_path}")

    print(
        f"Sample Size: {df.shape[0]} observations"
    )

    print(
        f"Variables Analyzed: {', '.join(list(df.columns))}"
    )

    print("\n" + "=" * 70 + "\n")

    # ---------------------------------------------------------------
    # REGRESSION MODELS
    # ---------------------------------------------------------------

    models_to_check = [

        # Model 1: Task-Technology Fit -> Productivity
        (
            ["TTF"],
            "Productivity",
            "Model 1: Task-Technology Fit -> Productivity"
        ),

        # Model 2: Task-Technology Fit -> Code Quality
        (
            ["TTF"],
            "Code_Quality",
            "Model 2: Task-Technology Fit -> Code Quality"
        ),

        # Model 3a: Job Demands -> Well-Being
        (
            ["Job_Demands"],
            "Well_Being",
            "Model 3a: Job Demands -> Well-Being"
        ),

        # Model 3b: Job Resources -> Well-Being
        (
            ["Job_Resources"],
            "Well_Being",
            "Model 3b: Job Resources -> Well-Being"
        ),

        # Model 3c: Job Demands + Job Resources -> Well-Being
        (
            ["Job_Demands", "Job_Resources"],
            "Well_Being",
            "Model 3c: Job Demands + Job Resources -> Well-Being"
        ),
    ]

    # ---------------------------------------------------------------
    # RUN BREUSCH-PAGAN TESTS
    # ---------------------------------------------------------------

    for predictors, outcome, label in models_to_check:

        # Check whether all required variables are present.
        missing = [
            column
            for column in predictors + [outcome]
            if column not in df.columns
        ]

        # Skip the model if required variables are missing.
        if missing:
            print(
                f"Model skipped - missing column(s): {missing}\n"
            )
            continue

        # Run the Breusch-Pagan test.
        check_model(
            df,
            predictors,
            outcome,
            label
        )

    # ---------------------------------------------------------------
    # SUMMARY FOOTER
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "END OF APPENDIX: BREUSCH-PAGAN HOMOSCEDASTICITY TEST RESULTS"
    )

    print("=" * 70)

    print(
        "\nNote: Models with p-values > 0.05 do not provide "
        "evidence of heteroscedasticity and therefore satisfy "
        "the homoscedasticity assumption."
    )

    print("=" * 70)


# ---------------------------------------------------------------
# RUN THE MAIN FUNCTION
# ---------------------------------------------------------------

if __name__ == "__main__":
    main()

