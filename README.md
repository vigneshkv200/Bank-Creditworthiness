# Credit Default Risk & Creditworthiness Modeling with CatBoost

An end-to-end machine learning pipeline built to assess customer creditworthiness and identify potential defaults (30+ Days Past Due) for a retail credit card portfolio.

## 📌 Business Problem
Bank GoodCredit aims to proactively control its credit default risk. The goal is to predict whether an existing credit card customer will fall into the **30+ DPD (Days Past Due)** risk bucket (`Bad_label = 1`). 

*   **Target:** `Bad_label` (0 = Good history, 1 = Bad history / 30+ DPD)
*   **Evaluation Metric:** **Gini Coefficient** (Project Benchmark: **37.9%**)

## 🗃️ Data Strategy & Architecture
The raw data is structured at a transactional/account level, containing multiple records per customer. To meet the credit risk standards, the data was consolidated into **one row per customer**.

*   `Cust_Account`: Historical credit card/loan account and repayment information.
*   `Cust_Enquiry`: Credit-seeking behavior and historical enquiry activity.
*   `Cust_Demographics`: Anonymized customer demographic application data.

## 🛠️ Key Feature Engineering & Cleaning
*   **Chronological Date Conversion:** Raw timestamps parsed and translated into elapsed durations (e.g., months since oldest/latest account opened, months since last payment).
*   **Robust Outlier Treatment:** Corrected negative current balances to prevent skewed utilization, and cleaned sentinel placeholder values (e.g., `999,999,999` in enquiry amounts) to `NaN`.
*   **Payment History Parsing:** Concatenated multi-part payment status fields and extracted monthly DPD status trends, percentage of months current, and time to first delinquency.

## 🚀 Model Development & Selection
We evaluated two core feature selection frameworks on the training split using a **5-Fold Stratified Cross-Validation** approach:
1.  **Model-Based Feature Importance (CatBoost)**
2.  **Information Value (IV) Analysis**

**Cross-Validation Gini Results:**
*   Model-based Top 40 features achieved a **Mean CV Gini of 35.38%**.
*   IV-based Top 80 features achieved a **Mean CV Gini of 34.52%**.
*   **Decision:** Selected the Top 40 Model-Based feature set for its superior generalization and compactness.

## 📊 Final Performance
*   **Holdout Test AUC:** `0.6830`
*   **Holdout Test Gini:** `36.61%` (Highly competitive, approaching the project target threshold)

### Risk Rank Ordering
Sorting the test set into risk deciles demonstrates strong rank-ordering capabilities:
*   **Decile 10 (Highest Risk):** Bad Rate of **7.74%** (Lift: **1.84x** over the average bad rate)
*   **Decile 1 (Lowest Risk):** Bad Rate of **0.42%**

## 💡 Key Business Insights
1.  **Credit Exposure:** Total credit limit (`total_credit_limit`) and peak utilized outstanding balance (`max_cur_balance_amt`) are the strongest predictors of default risk.
2.  **Credit-Seeking Intensity:** Recent enquiry velocity (`count_enquiry_recency_90`) indicates active credit hunger and strongly correlates with higher default rates.
3.  **Repayment Behavior:** Established payment longevity and a clean monthly payment history string are critical markers of lower-risk profiles.
