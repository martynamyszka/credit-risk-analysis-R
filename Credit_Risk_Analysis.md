Credit Risk Predictive Modeling using Decision Trees and AdaBoost
================
Martyna Myszka
2026-10-11

## Business Problem & Objective

The goal of this project is to identify credit applicants with a high
risk of default. Using historical data from a German credit agency (1000
loan applicants), we aim to build a predictive model that helps the bank
minimize financial losses by rejecting high-risk applications.

## Data Loading and Preprocessing

First, we load the dataset and prepare our target variable ‘default’.

``` r
library(readr)
library(ggplot2)
library(C50)
library(gmodels)
dane <- read.csv("credit.csv", sep = ";", header = TRUE, stringsAsFactors = TRUE)
head(dane)
```

    ##   checking_balance months_loan_duration credit_history              purpose
    ## 1           < 0 DM                    6       critical furniture/appliances
    ## 2       1 - 200 DM                   48           good furniture/appliances
    ## 3          unknown                   12       critical            education
    ## 4           < 0 DM                   42           good furniture/appliances
    ## 5           < 0 DM                   24           poor                  car
    ## 6          unknown                   36           good            education
    ##   amount savings_balance employment_duration percent_of_income
    ## 1   1169         unknown           > 7 years                 4
    ## 2   5951        < 100 DM         1 - 4 years                 2
    ## 3   2096        < 100 DM         4 - 7 years                 2
    ## 4   7882        < 100 DM         4 - 7 years                 2
    ## 5   4870        < 100 DM         1 - 4 years                 3
    ## 6   9055         unknown         1 - 4 years                 2
    ##   years_at_residence age other_credit housing existing_loans_count       job
    ## 1                  4  67         none     own                    2   skilled
    ## 2                  2  22         none     own                    1   skilled
    ## 3                  3  49         none     own                    1 unskilled
    ## 4                  4  45         none   other                    1   skilled
    ## 5                  4  53         none   other                    2   skilled
    ## 6                  4  35         none   other                    1 unskilled
    ##   dependents phone default
    ## 1          1   yes      no
    ## 2          1    no     yes
    ## 3          2    no      no
    ## 4          2    no      no
    ## 5          2    no     yes
    ## 6          2   yes      no

``` r
dane$default <- as.factor(dane$default)
```

## Exploratory Data Analysis (EDA)

Before modeling, we explore some key variables to understand what drives
loan defaults.

``` r
# Loan Amount Distribution
ggplot(dane, aes(x = amount, fill = default)) +
  geom_histogram(bins = 30, color = "black", alpha = 0.6, position = "identity") +
  theme_minimal() +
  labs(title = "Loan Amount Distribution", x = "Amount", fill = "Default")
```

![](https://github.com/martynamyszka/credit-risk-analysis-R/blob/main/unnamed-chunk-1-1.png?raw=true)<!-- -->

### Conclusions from the ‘amount’ variable distribution analysis

- **Right-skewed distribution:** Small loan amounts dominate (peak
  around 1000-3500).
- **Low risk indicator:** Small amounts indicate a low risk of default
  (a vast majority of loans are successfully repaid).
- **Higher risk threshold:** Large amounts (e.g., \>7500) are associated
  with higher risk (the proportion of defaults increases).
- **Summary:** The ‘amount’ feature effectively differentiates risk and
  will be useful in the model.

``` r
# Loan Duration Distribution
ggplot(dane, aes(x = months_loan_duration, fill = default)) +
  geom_histogram(bins = 20, color = "black", alpha = 0.6, position = "identity") +
  theme_minimal() +
  labs(title = "Loan Duration Distribution", x = "Months", fill = "Default")
```

![](https://github.com/martynamyszka/credit-risk-analysis-R/blob/main/unnamed-chunk-2-1.png?raw=true)<!-- -->

### Conclusions from the ‘months_loan_duration’ variable distribution analysis:

- Most loans in the dataset are short-to-medium term (concentrated
  between 6 and 24 months).
- **Risk trend:** The longer the loan duration, the higher the relative
  risk of default.
- **High-risk segment:** Loans lasting more than 36 months show a
  critically high default rate (the proportion of defaults often equals
  or exceeds successful repayments).
- **Summary:** Loan duration is a very strong indicator of credit risk
  and will be an important feature in the predictive model.

Let’s also check how the distribution of some categorical variable looks
like.

``` r
# 3. Checking balance distribution
ggplot(dane, aes(x = checking_balance, fill = default)) +
  geom_bar(position = "dodge", color = "black") +
  theme_minimal() +
  labs(title = "Checking balance Distribution", 
       x = "Checking balance", 
       fill ="Default") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

![](https://github.com/martynamyszka/credit-risk-analysis-R/blob/main/unnamed-chunk-3-1.png?raw=true)<!-- -->

### Conclusions from the ‘checking_balance’ (categorical variable) distribution analysis:

- **Critical risk:** Applicants with a negative balance (‘\< 0 DM’) have
  an extremely high default rate, with almost equal numbers of repaid
  and defaulted loans.
- **Moderate risk:** Customers with a low positive balance (‘1 - 200
  DM’) show a noticeable default rate, though the majority still repay.
- **Safe segments:** High balances (‘\> 200 DM’) and ‘unknown’ statuses
  indicate very low credit risk.
- **Business Insight:** Interestingly, the ‘unknown’ category contains
  the largest number of applicants and represents the safest borrower
  profile. This variable strongly separates risk profiles.

### Overall Default Rate (Baseline)

Before building the predictive model, it is crucial to establish the
baseline risk by checking the overall proportion of defaulted loans in
the dataset. This helps us understand the class balance and sets a
performance benchmark for our algorithms.

``` r
# Method 1: Generating a proportion table for all classes
prop.table(table(dane$default))
```

    ## 
    ##  no yes 
    ## 0.7 0.3

``` r
# Method 2: Quick calculation of the exact default rate (fraction of 'yes')
# R evaluates 'yes' as 1 and 'no' as 0, so the mean gives the exact probability
mean(dane$default == "yes")
```

    ## [1] 0.3

## Train/Test Data Split

We split the data into a training set (90%) and a testing set (10%) to
evaluate our model on unseen data. We don’t know if the data is ordered
in a specific way, so the best solution here is to randomize its order.

``` r
RNGversion("3.5.2")
```

    ## Warning in RNGkind("Mersenne-Twister", "Inversion", "Rounding"): non-uniform
    ## 'Rounding' sampler used

``` r
set.seed(123)
#now we choose randomly 900 indices of first 1000
sample_train=sample(1000,900)
#str(sample_train)
zbior_uczacy=dane[sample_train,]
zbior_testowy=dane[-sample_train,] #here by using '-sample_train' we take the indices that weren't sampled above
```

Now let’s check if the proportions of the ‘default’ class are similar in
both sets. This is important because a bad random split can ruin the
model’s performance. If the difference is too big, we would need to
adjust our sampling method.

``` r
prop.table(table(zbior_testowy$default))
```

    ## 
    ##   no  yes 
    ## 0.67 0.33

``` r
prop.table(table(zbior_uczacy$default))
```

    ## 
    ##        no       yes 
    ## 0.7033333 0.2966667

``` r
#they are similar, so we can continue
```

## Model Training: C5.0 Decision Tree

We build a baseline decision tree model to predict the default status.

``` r
model <- C5.0(zbior_uczacy[-17], zbior_uczacy$default, trials = 1, costs = NULL) #[-17] means that 'default' is gone from it
model
```

    ## 
    ## Call:
    ## C5.0.default(x = zbior_uczacy[-17], y = zbior_uczacy$default, trials = 1,
    ##  costs = NULL)
    ## 
    ## Classification Tree
    ## Number of samples: 900 
    ## Number of predictors: 16 
    ## 
    ## Tree size: 57 
    ## 
    ## Non-standard options: attempt to group attributes

Let’s look at the tree structure:

``` r
summary(model)
```

    ## 
    ## Call:
    ## C5.0.default(x = zbior_uczacy[-17], y = zbior_uczacy$default, trials = 1,
    ##  costs = NULL)
    ## 
    ## 
    ## C5.0 [Release 2.07 GPL Edition]      Mon Oct 12 15:09:09 2026
    ## -------------------------------
    ## 
    ## Class specified by attribute `outcome'
    ## 
    ## Read 900 cases (17 attributes) from undefined.data
    ## 
    ## Decision tree:
    ## 
    ## checking_balance in {> 200 DM,unknown}: no (412/50)
    ## checking_balance in {< 0 DM,1 - 200 DM}:
    ## :...credit_history in {perfect,very good}: yes (59/18)
    ##     credit_history in {critical,good,poor}:
    ##     :...months_loan_duration <= 22:
    ##         :...credit_history = critical: no (72/14)
    ##         :   credit_history = poor:
    ##         :   :...dependents > 1: no (5)
    ##         :   :   dependents <= 1:
    ##         :   :   :...years_at_residence <= 3: yes (4/1)
    ##         :   :       years_at_residence > 3: no (5/1)
    ##         :   credit_history = good:
    ##         :   :...savings_balance in {> 1000 DM,500 - 1000 DM}: no (15/1)
    ##         :       savings_balance = 100 - 500 DM:
    ##         :       :...other_credit = bank: yes (3)
    ##         :       :   other_credit in {none,store}: no (9/2)
    ##         :       savings_balance = unknown:
    ##         :       :...other_credit = bank: yes (1)
    ##         :       :   other_credit in {none,store}: no (21/8)
    ##         :       savings_balance = < 100 DM:
    ##         :       :...purpose in {business,car0,renovations}: no (8/2)
    ##         :           purpose = education:
    ##         :           :...checking_balance = < 0 DM: yes (4)
    ##         :           :   checking_balance = 1 - 200 DM: no (1)
    ##         :           purpose = car:
    ##         :           :...employment_duration = > 7 years: yes (5)
    ##         :           :   employment_duration = unemployed: no (4/1)
    ##         :           :   employment_duration = < 1 year:
    ##         :           :   :...years_at_residence <= 2: yes (5)
    ##         :           :   :   years_at_residence > 2: no (3/1)
    ##         :           :   employment_duration = 1 - 4 years:
    ##         :           :   :...years_at_residence <= 2: yes (2)
    ##         :           :   :   years_at_residence > 2: no (6/1)
    ##         :           :   employment_duration = 4 - 7 years:
    ##         :           :   :...amount <= 1680: yes (2)
    ##         :           :       amount > 1680: no (3)
    ##         :           purpose = furniture/appliances:
    ##         :           :...job in {management,unskilled}: no (23/3)
    ##         :               job = unemployed: yes (1)
    ##         :               job = skilled:
    ##         :               :...months_loan_duration > 13: [S1]
    ##         :                   months_loan_duration <= 13:
    ##         :                   :...housing in {other,own}: no (23/4)
    ##         :                       housing = rent:
    ##         :                       :...percent_of_income <= 3: yes (3)
    ##         :                           percent_of_income > 3: no (2)
    ##         months_loan_duration > 22:
    ##         :...savings_balance = > 1000 DM: no (2)
    ##             savings_balance = 500 - 1000 DM: yes (4/1)
    ##             savings_balance = 100 - 500 DM:
    ##             :...credit_history in {critical,poor}: no (14/3)
    ##             :   credit_history = good:
    ##             :   :...other_credit = bank: no (1)
    ##             :       other_credit in {none,store}: yes (12/2)
    ##             savings_balance = unknown:
    ##             :...checking_balance = 1 - 200 DM: no (17)
    ##             :   checking_balance = < 0 DM:
    ##             :   :...credit_history = critical: no (1)
    ##             :       credit_history in {good,poor}: yes (12/3)
    ##             savings_balance = < 100 DM:
    ##             :...months_loan_duration > 47: yes (21/2)
    ##                 months_loan_duration <= 47:
    ##                 :...housing = other:
    ##                     :...percent_of_income <= 2: no (6)
    ##                     :   percent_of_income > 2: yes (9/3)
    ##                     housing = rent:
    ##                     :...other_credit = bank: no (1)
    ##                     :   other_credit in {none,store}: yes (16/3)
    ##                     housing = own:
    ##                     :...employment_duration = > 7 years: no (13/4)
    ##                         employment_duration = 4 - 7 years:
    ##                         :...job in {management,skilled,
    ##                         :   :       unemployed}: yes (9/1)
    ##                         :   job = unskilled: no (1)
    ##                         employment_duration = unemployed:
    ##                         :...years_at_residence <= 2: yes (4)
    ##                         :   years_at_residence > 2: no (3)
    ##                         employment_duration = 1 - 4 years:
    ##                         :...purpose in {business,car0,education}: yes (7/1)
    ##                         :   purpose in {furniture/appliances,
    ##                         :   :           renovations}: no (7)
    ##                         :   purpose = car:
    ##                         :   :...years_at_residence <= 3: yes (3)
    ##                         :       years_at_residence > 3: no (3)
    ##                         employment_duration = < 1 year:
    ##                         :...years_at_residence > 3: yes (5)
    ##                             years_at_residence <= 3:
    ##                             :...other_credit = bank: no (0)
    ##                                 other_credit = store: yes (1)
    ##                                 other_credit = none:
    ##                                 :...checking_balance = 1 - 200 DM: no (8/2)
    ##                                     checking_balance = < 0 DM:
    ##                                     :...job in {management,skilled,
    ##                                         :       unemployed}: yes (2)
    ##                                         job = unskilled: no (3/1)
    ## 
    ## SubTree [S1]
    ## 
    ## employment_duration in {< 1 year,4 - 7 years}: no (4)
    ## employment_duration in {> 7 years,1 - 4 years,unemployed}: yes (10)
    ## 
    ## 
    ## Evaluation on training data (900 cases):
    ## 
    ##      Decision Tree   
    ##    ----------------  
    ##    Size      Errors  
    ## 
    ##      56  133(14.8%)   <<
    ## 
    ## 
    ##     (a)   (b)    <-classified as
    ##    ----  ----
    ##     598    35    (a): class no
    ##      98   169    (b): class yes
    ## 
    ## 
    ##  Attribute usage:
    ## 
    ##  100.00% checking_balance
    ##   54.22% credit_history
    ##   47.67% months_loan_duration
    ##   38.11% savings_balance
    ##   14.33% purpose
    ##   14.33% housing
    ##   12.56% employment_duration
    ##    9.00% job
    ##    8.67% other_credit
    ##    6.33% years_at_residence
    ##    2.22% percent_of_income
    ##    1.56% dependents
    ##    0.56% amount
    ## 
    ## 
    ## Time: 0.0 secs

### Decision Tree Logic

The generated C5.0 tree reveals the primary rules the model uses to
classify applicants:

- **The Root Node (Most Important Feature):** ‘checking_balance’ is the
  absolute most important variable (100% attribute usage). If an
  applicant has a balance ‘\> 200 DM’ or an ‘unknown’ account, the model
  instantly classifies them as low-risk (class ‘no’), terminating the
  decision path for 412 applicants.
- **The “Trap” of Good Credit History:** Interestingly, for applicants
  with poor checking balances (‘\< 0 DM’ or ‘1 - 200 DM’), having a
  ‘perfect’ or ‘very good’ credit history leads straight to a high-risk
  classification (class ‘yes’). *Business insight: This might indicate
  applicants who repay other small loans flawlessly but are now taking a
  loan they cannot afford.*
- **Loan Duration & Savings:** For the remaining applicants, the model
  splits them primarily by ‘months_loan_duration’ (the threshold is 22
  months). Long loans combined with low savings (‘savings_balance \< 100
  DM’) drastically increase the predicted risk of default.

## Model Evaluation

We test the model against our test set.

``` r
credit_prediction <- predict(model, zbior_testowy)
CrossTable(credit_prediction,zbior_testowy$default,prop.c = FALSE,prop.chisq = FALSE,prop.r=FALSE,
           dnn=c('predicted default', 'actual default'))
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                   | actual default 
    ## predicted default |        no |       yes | Row Total | 
    ## ------------------|-----------|-----------|-----------|
    ##                no |        59 |        19 |        78 | 
    ##                   |     0.590 |     0.190 |           | 
    ## ------------------|-----------|-----------|-----------|
    ##               yes |         8 |        14 |        22 | 
    ##                   |     0.080 |     0.140 |           | 
    ## ------------------|-----------|-----------|-----------|
    ##      Column Total |        67 |        33 |       100 | 
    ## ------------------|-----------|-----------|-----------|
    ## 
    ## 

As shown in the CrossTable above, the model correctly predicted 73% of
the cases. However, it misclassifies too many risky loans as safe. This
is a very costly error for a bank.

### Model Optimization: AdaBoost

To reduce the costly misclassifications, we apply the Adaptive Boosting
(AdaBoost) algorithm with 10 trials. The model will focus on the most
difficult cases from the previous iterations.

``` r
credit_boost10=C5.0(zbior_uczacy[-17],zbior_uczacy$default,trials=10)
credit_boost10
```

    ## 
    ## Call:
    ## C5.0.default(x = zbior_uczacy[-17], y = zbior_uczacy$default, trials = 10)
    ## 
    ## Classification Tree
    ## Number of samples: 900 
    ## Number of predictors: 16 
    ## 
    ## Number of boosting iterations: 10 
    ## Average tree size: 47.5 
    ## 
    ## Non-standard options: attempt to group attributes

``` r
#summary(credit_boost10)

predykcja_boost=predict(credit_boost10,zbior_testowy)
CrossTable(predykcja_boost,zbior_testowy$default,
           prop.c=FALSE, prop.chisq = FALSE, prop.r=FALSE,
           dnn=c('predicted','actual'))
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##              | actual 
    ##    predicted |        no |       yes | Row Total | 
    ## -------------|-----------|-----------|-----------|
    ##           no |        62 |        13 |        75 | 
    ##              |     0.620 |     0.130 |           | 
    ## -------------|-----------|-----------|-----------|
    ##          yes |         5 |        20 |        25 | 
    ##              |     0.050 |     0.200 |           | 
    ## -------------|-----------|-----------|-----------|
    ## Column Total |        67 |        33 |       100 | 
    ## -------------|-----------|-----------|-----------|
    ## 
    ## 

The results after using the AdaBoost method are much better. 

## Final Business Conclusions 
Applying AdaBoost significantly reduced the total error rate. 
While predicting defaults remains a challenging task, the
ensemble model outperforms the single decision tree, providing a more
robust risk management tool for the financial institution.
