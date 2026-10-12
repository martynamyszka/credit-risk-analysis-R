# Credit Risk Analysis & Prediction 📊

## Project Overview
This is an end-to-end data analysis and machine learning project focused on predicting credit default risk. Using the classic German Credit dataset, the project demonstrates a complete analytical workflow: from Exploratory Data Analysis (EDA) and baseline establishment to building and optimizing decision tree models.

The primary business objective of this analysis is not just overall accuracy, but specifically minimizing **False Negatives** — the costly error of classifying a high-risk applicant as safe, which results in significant financial loss for a bank.

## Tech Stack & Tools
* **Language:** R
* **Environment:** RStudio / R Markdown
* **Data Manipulation & Visualization:** Data frames, `ggplot2`
* **Machine Learning:** `C50` (Decision Trees), `gmodels` (CrossTable evaluation), AdaBoost ensemble method

## Key Business Insights (EDA)
Before modeling, the data revealed several crucial risk indicators:
* **The "Perfect History" Paradox:** Applicants with "perfect" credit histories often showed higher default rates, likely indicating individuals with no prior credit footprint (a clean slate) rather than proven reliable borrowers.
* **Checking Balance:** This proved to be the strongest predictor. Applicants with negative balances (< 0 DM) present a critical risk, while those with 'unknown' balances represent the safest segment.
* **Loan Duration & Amount:** A clear trend shows that longer loans (especially >36 months) and larger amounts (>7500) significantly increase the relative risk of default.

## Modeling Approach
1. **Baseline Assessment:** Established initial class balance to understand the underlying default rate and set a benchmark.
2. **Train/Test Split:** Randomized 90/10 split to evaluate the model on unseen data and verify class stratification.
3. **Base Model (C5.0):** Built an initial decision tree to extract core business rules and evaluate the baseline confusion matrix.
4. **Optimization (AdaBoost):** Applied adaptive boosting to improve the model's ability to identify complex risk patterns and reduce the number of costly misclassifications.

## How to view the project
To see the full code, visualizations, and detailed business commentary, simply open the `Credit_Risk_Analysis.md` (or `.html`) file in this repository.
