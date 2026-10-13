# Credit Risk Analysis & Prediction 

## Project Overview
This repository contains a machine learning model built in R to predict credit default risk. Using the German Credit dataset, I wanted to see how well we can predict loan defaults based on financial history. 

The main goal of this project wasn't just to maximize overall accuracy. From a banking perspective, predicting a bad loan as a good one (False Negative) is the most expensive mistake. Therefore, the focus here is heavily on minimizing those specific errors.

## Tech Stack
* **Language:** R
* **Environment:** RStudio / R Markdown
* **Libraries:** `ggplot2` (visualization), `C50` (Decision Trees), `gmodels` (evaluation)

## Key Findings from EDA
Before building the model, data exploration revealed a few interesting patterns:
* **No history isn't good history:** Applicants with "perfect" credit histories actually defaulted more often. This likely means they simply have no credit footprint at all, rather than a proven track record.
* **Checking balances matter most:** Having a negative balance is the strongest predictor of default, while 'unknown' balances are surprisingly safe.
* **Size and length:** As expected, longer loans (>36 months) and larger amounts (>7500 DM) carry a significantly higher relative risk.

## Modeling Steps
1. **Baseline & Split:** Checked the initial default rate and split the data (90/10) to evaluate on unseen data.
2. **Base Model (C5.0):** Trained a simple decision tree to see the initial rules and check the baseline confusion matrix.
3. **Ensemble (AdaBoost):** Applied adaptive boosting to improve the model's performance and reduce the number of costly False Negatives.

## How to view the project
Simply open the `Credit_Risk_Analysis.md` file in this repository to see the full code, plots, and my step-by-step commentary.


