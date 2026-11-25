# Credit Card Default Risk & Creditworthiness Modeling with CatBoost

This repository hosts an end-to-end machine learning pipeline built to assess retail customer creditworthiness and predict the probability of a customer falling into a **30+ DPD (Days Past Due)** default risk bucket (`Bad_label = 1`) for a credit card portfolio.

## 📌 Business Problem & Objectives
Bank GoodCredit aims to proactively manage credit exposure and minimize financial defaults while optimizing card approvals. 
- **Target variable (`Bad_label`):** `0` (Good credit history) or `1` (Bad credit history / 30+ DPD).
- **Target Imbalance:** Highly imbalanced data containing only **4.2% bad cases**.
- **Primary Metric:** **Gini Coefficient** (Project Benchmark: **37.9% Gini**).

## 📂 Data Consolidation & Pipeline Architecture
The raw bureau and transaction data is distributed across three source tables with multiple transaction/history logs per customer. To meet risk management modeling standards, we consolidated this into a **single analytical table containing exactly one row per customer**.

1.  **`Cust_Demographics` (Spine):** Application-level demographic data (1 row per customer).
2.  **`Cust_Account` (Consolidated):** Multiple rows per customer depicting historical tradeline details, limits, and payments.
3.  **`Cust_Enquiry` (Consolidated):** Multiple rows per customer showing historical credit-seeking behavior (enquiries).

## 🛠️ Advanced Feature Engineering & Cleaning
*   **Timeline Alignment (Reference Anchor):** The application date (`dt_opened`) was verified as constant per customer and used as a chronological "anchor" to compute absolute recency/gap features in months.
*   **Payment History Parsing:** Concatenated the split history strings (`paymenthistory1` and `paymenthistory2`) into a unified string. Parsed monthly 3-character bureau status codes into percentages of months spent in distinct DPD buckets (`0-29`, `30-59`, `60-89`, `90+` DPD) and months elapsed to first delinquency.
*   **Outlier & Sentinel Neutralization:** Identified and cleaned extreme database sentinel placeholders (e.g., `999,999,999` in enquiry amounts) and corrected negative outstanding current balances to `NaN` to prevent distorting aggregate risk statistics.
*   **Financial Ratios:** Engineered utilization metrics such as `Ratio_currbalance_creditlimit` and a dynamic capacity trend measure (`utilisation_trend`).

## 🤖 Why CatBoost Classifier?
We selected **CatBoost** over other gradient boosted tree frameworks (e.g., XGBoost, LightGBM) for several key reasons:
1.  **Native Categorical Handling:** CatBoost robustly builds target statistics on categorical columns natively, eliminating the risk of high-cardinality dimensionality explosion associated with extensive One-Hot Encoding.
2.  **Symmetric Trees:** Symmetric trees act as a regularizer, significantly reducing overfitting tendencies on highly imbalanced targets.
3.  **Built-In Missing Value Handling:** It natively handles missing numeric feature fields (such as unreported interest rates or limits) without needing arbitrary imputation strategies.

## 🔬 Feature Selection & 5-Fold Cross-Validation
To construct a highly predictive, parsimonious model, we evaluated two distinct feature-selection strategies using **5-Fold Stratified Cross-Validation** (to preserve the 4.2% default imbalance across folds) on the training set:

### 1. Model-Based Feature Importance (CatBoost Gain)
We iteratively tested various subsets based on the model's gain output. The **Top 40 features** yielded the highest performance:
*   **Mean CV Gini:** **35.38%** (Std CV Gini: 2.67%)
*   **Out-of-Fold (OOF) Gini:** **35.31%**

### 2. Information Value (IV) Feature Selection
We calculated the Information Value for all variables and tested corresponding Top-N feature counts. The **Top 80 features** yielded the best IV configuration:
*   **Mean CV Gini:** **34.52%** (Std CV Gini: 2.28%)
*   **Out-of-Fold (OOF) Gini:** **34.46%**

### Decision
The **Top 40 Model-Based feature set** was selected for final model training because it delivered a superior Gini performance with half the dimensionality of the IV approach.

## 📊 Final Performance Metrics
*   **Holdout Test AUC:** `0.6830`
*   **Holdout Test Gini:** `36.61%` (Extremely close to the 37.90% benchmark, demonstrating high model generalization without overfitting).

### Risk Rank-Ordering (Decile Table)
By sorting customers on predicted risk probability and partitioning into ten equal-sized deciles, the model displays robust risk-sorting capabilities:

| Decile | Customers | Bad Customers | Bad Rate | Lift |
| :---: | :---: | :---: | :---: | :---: |
| **10 (Highest Risk)** | 478 | 37 | **7.74%** | **1.84x** |
| **9** | 478 | 37 | **7.74%** | **1.84x** |
| **8** | 478 | 37 | **7.74%** | **1.84x** |
| **7** | 478 | 31 | 6.49% | 1.54x |
| **6** | 478 | 14 | 2.93% | 0.70x |
| **5** | 478 | 12 | 2.51% | 0.60x |
| **4** | 478 | 12 | 2.51% | 0.60x |
| **3** | 478 | 8  | 1.67% | 0.40x |
| **2** | 478 | 11 | 2.30% | 0.55x |
| **1 (Lowest Risk)** | 478 | 2  | **0.42%** | **0.10x** |

## 💡 Key Business & Risk Insights
1.  **Credit Utilization & Exposure:** Total credit limit (`total_credit_limit`) and peak outstanding balance (`max_cur_balance_amt`) are the highest-weighted features in separating default profiles, showcasing risk aggregation.
2.  **Credit-Seeking Intensity:** Recent enquiry velocity within 90 days (`count_enquiry_recency_90`) indicates active credit hunger, correlating strongly with a high probability of delinquency.
3.  **Established Payment History:** A clean historical record with a high proportion of months spent in the `0-29 DPD` bucket serves as an essential anchor of low-risk creditworthiness.
```
