
<img width="1100" height="220" alt="pic" src="https://github.com/user-attachments/assets/f4505bd2-1de1-4609-b0c5-2b5cf4307230" />

## 🎯 Objective

<img width="1600" height="1000" alt="pic" src="https://github.com/user-attachments/assets/8a823a6c-6c08-4207-9daf-db819c2df717" />






## ♻️ WorkFlow :-

<img width="1500" height="760" alt="pic" src="https://github.com/user-attachments/assets/ec2077fa-7ef7-4d07-8b6a-0132a4e0752e" />




## 📂 Project Files
| 📄 File | 📌 Description |
|---|---|
| `main_transactions.csv` | Raw transaction-level data (customer_id, income, loan, credit score, default_flag) |
| `customer_metadata.json` | Raw customer demographic metadata |
| `credit_risk.db` | SQLite database with `loan_repayment_history` table |
| `Customer_Credit_Risk_Analysis_with_Insights.ipynb` | Main notebook — acquisition, cleaning, outliers, feature engineering |
| `customer_credit_risk_merged_raw.csv` | Checkpoint after merging all 4 sources |
| `customer_credit_risk_profile.html` | Auto-generated YData Profiling data-quality report |
| `final_customer_credit_risk_dataset.csv` | Final ML-ready dataset |

## 🛠️ Tools Used

![Python](https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/JUPYTER-4D4D4D?style=for-the-badge&logo=jupyter&logoColor=white)
![Notebook](https://img.shields.io/badge/NOTEBOOK-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/PANDAS-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NUMPY-013243?style=for-the-badge&logo=numpy&logoColor=white)
![JSON](https://img.shields.io/badge/JSON-000000?style=for-the-badge&logo=json&logoColor=white)
![SQLite3](https://img.shields.io/badge/SQLITE3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Requests](https://img.shields.io/badge/REQUESTS-F7931E?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/SCIKIT--LEARN-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/MATPLOTLIB-2E2A5B?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/SEABORN-388E3C?style=for-the-badge&logo=python&logoColor=white)
![YData Profiling](https://img.shields.io/badge/YDATA%20PROFILING-EC407A?style=for-the-badge&logo=databricks&logoColor=white)


---

## 🎬 Project Demo

[![Watch Demo](https://img.shields.io/badge/Watch%20Demo-Google%20Drive-blue?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1215xHkwTyCsBinIyIcUpixhc1skNKj2z/view?usp=sharing)

📹 Click the badge above to watch the complete project demonstration.


## 🧬 Dataset Structure

| Source | Shape | Join Key |
|---|---|---|
| Transactions | 1000 × 9 | `customer_id` |
| Customer Metadata | 1000 × 6 | `customer_id` |
| Repayment History | 1008 × 2 | `customer_id` |
| Economic Data | 4 × 4 | `region` |
| **Final Merged** | **1008 × 18** | — |

---

## 📥 Part B: Data Acquisition & Merging


<img width="1100" height="220" alt="pic" src="https://github.com/user-attachments/assets/acf585c3-027d-47b7-bbe6-66050103d7b9" />




Loaded four different formats — CSV, JSON, SQL, and API — and merged them into one table.

```python
transactions = pd.read_csv("main_transactions.csv")
customer_metadata = pd.DataFrame(json.load(open("customer_metadata.json")))

connection = sqlite3.connect("credit_risk.db")
repayment_history = pd.read_sql("SELECT * FROM loan_repayment_history", connection)

# Dummy API: regional economic indicators
economic_data = pd.DataFrame(api_data["data"],
    columns=["region", "inflation_rate", "unemployment_rate", "economic_indicator_date"])

data = transactions.merge(customer_metadata, on="customer_id", how="left")
data = data.merge(repayment_history, on="customer_id", how="left")
data = data.merge(economic_data, on="region", how="left")
```

💡 **Insight:** Final merged dataset: **1008 × 18**. Row count rose from 1000 → 1008 because `loan_repayment_history` had duplicate `customer_id` entries. Post-merge, `annual_income` (70), `credit_score` (62), `age` (51), `gender` (60), and `employment_type` (51) had missing values (~5–7%).

---

## 🧼 Part C: Data Understanding & Cleaning


<img width="1100" height="220" alt="pic" src="https://github.com/user-attachments/assets/11f03109-fa8a-4942-91dc-463056ab1cb0" />





Compared **five** missing-value strategies before picking the best one for each column type.

```python
# Simple Imputer — mean vs median
data_mean[numeric_columns]   = SimpleImputer(strategy="mean").fit_transform(...)
data_median[numeric_columns] = SimpleImputer(strategy="median").fit_transform(...)

# Categorical — most frequent
data_simple[categorical_columns] = SimpleImputer(strategy="most_frequent", missing_values=None).fit_transform(...)

# Missing Indicator + Random Sample Imputation
data_indicator["annual_income_missing"] = data_indicator["annual_income"].isnull().astype(int)

# KNN Imputer (multivariate)
data_knn[knn_columns] = KNNImputer(n_neighbors=5).fit_transform(...)

# MICE (Iterative Imputer)
data_mice[mice_columns] = IterativeImputer(random_state=42, max_iter=10).fit_transform(...)

# Complete Case Analysis
data_complete = data.dropna()   # 1008 → 746 rows (-26%)
```

💡 **Insight:** `annual_income`'s mean (₹8.98L) sits well above its median (₹6.72L) — a clear right-skew signal, so **median** was preferred over mean for it. Complete Case Analysis dropped **262 rows (~26%)** — too costly, so it was ruled out. **KNN and MICE** were chosen as the most reliable imputers since they use feature relationships instead of a single statistic; both brought missing counts to **zero**.

---

## 🎯 Part D: Outlier Handling


<img width="1100" height="220" alt="pic" src="https://github.com/user-attachments/assets/3a66035c-a3a3-4551-966e-4e2e14976b98" />


Compared **three** detection methods and one non-destructive treatment on `annual_income`, `loan_amount`, `credit_score`.

```python
# Z-score (threshold = 3)
z_scores = zscore(data[numeric_columns], nan_policy="omit")
z_outliers = (z_scores.abs() > 3)              # 26 total

# IQR
Q1, Q3 = data[col].quantile([0.25, 0.75]); IQR = Q3 - Q1
iqr_outliers = (data[col] < Q1-1.5*IQR) | (data[col] > Q3+1.5*IQR)   # 112 total

# 1st–99th Percentile
percentile_outliers = (data[col] < data[col].quantile(0.01)) | (data[col] > data[col].quantile(0.99))  # 51 total

# Winsorization (non-destructive)
data_winsor[col] = winsorize(data_winsor[col], limits=[0.01, 0.01], nan_policy="omit")
```

💡 **Insight — Method comparison:** IQR flagged the most outliers (**112**), Percentile was balanced (**51**), Z-score was least sensitive (**26**). Row-drop impact: Z-score removed 17 rows, Percentile 42, IQR 82 (the most aggressive). **Winsorization** was chosen as the final treatment since it **caps instead of deletes** — `annual_income`'s minimum rose from ₹1.2L to ₹2.02L, with the full **1008 rows preserved**.

---

## 🛠️ Part E: Feature Engineering



<img width="1100" height="220" alt="pic" src="https://github.com/user-attachments/assets/7528071a-8a55-44a5-b6bd-5b3ac9991ba3" />


**Variable types:** 7 numeric (age, annual_income, loan_amount, credit_score, repayment_history, transaction_count, spending_ratio) + 5 categorical (gender, region, education_level, employment_type, loan_purpose).

```python
# Date features
data_date["join_year"]    = data_date["join_date"].dt.year
data_date["join_month"]   = data_date["join_date"].dt.month
data_date["join_weekday"] = data_date["join_date"].dt.day_name()

# Encoding: Ordinal (education), Label (gender), One-Hot (region, loan_purpose)
ordinal_encoder = OrdinalEncoder(categories=[["Primary","Secondary","Graduate","Post-Graduate"]])
data_label["gender_encoded"] = LabelEncoder().fit_transform(data_label["gender"].fillna("Unknown"))
data_onehot = pd.get_dummies(data, columns=["region", "loan_purpose"], dtype=int)

# Numeric binning: equal-width, quantile, K-Means
data_binning["income_group"]         = pd.cut(data["annual_income"], bins=4, labels=[...])
data_quantile["transaction_quantile"] = pd.qcut(data["transaction_count"], q=4, labels=[...])
data_kmeans["transaction_cluster"]    = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(...)
```

💡 **Insight:** `education_level` ordinally encoded (Primary=0 < ... < Post-Graduate=3) to preserve its natural order; `region`/`loan_purpose` one-hot encoded (no natural order) — column count grew 18 → 25. Equal-width binning skewed most customers into 'Low' income due to extreme outliers, while quantile binning kept groups balanced — showing why **quantile/cluster-based binning beats fixed-width on skewed data**. `credit_score > 700` flag: 660 below vs 348 above — majority sit in the lower credit tier.

---

## ⚖️ Part F: Feature Scaling


<img width="1100" height="220" alt="pic" src="https://github.com/user-attachments/assets/61dd25e8-22e8-4c6c-ad33-f5555091660b" />

Compared **five** scalers side by side on the 7 numeric columns (after median imputation).

```python
StandardScaler().fit_transform(scaling_data[numeric_columns])   # mean≈0, std≈1
Normalizer().fit_transform(scaling_data[numeric_columns])       # row-wise unit norm
MinMaxScaler().fit_transform(scaling_data[numeric_columns])     # range [0,1]
MaxAbsScaler().fit_transform(scaling_data[numeric_columns])     # range [-1,1], no centering
RobustScaler().fit_transform(scaling_data[numeric_columns])     # median/IQR based
```

💡 **Insight:** All five scalers produce noticeably different ranges. Given the outliers/skewness in `annual_income` and `loan_amount`, **RobustScaler** (median + IQR based) was chosen as the most stable — StandardScaler and MinMaxScaler are both more prone to distortion from extreme values.

---

## 🏗️ Part G: Feature Construction & Transformation


<img width="1100" height="220" alt="pic" src="https://github.com/user-attachments/assets/453290fa-11de-4ad6-8059-774bbe29ada3" />



```python
# Skew-correcting transforms
transform_data["spending_ratio_log"] = np.log1p(transform_data["spending_ratio"])
power_data["annual_income_boxcox"]   = PowerTransformer(method="box-cox").fit_transform(power_data[["annual_income"]])
power_data["loan_amount_yeojohnson"] = PowerTransformer(method="yeo-johnson").fit_transform(power_data[["loan_amount"]])

# ColumnTransformer pipeline: numeric (impute+scale) + categorical (impute+one-hot)
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", missing_values=None)),
                               ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features)
])

# New business features
feature_data["debt_to_income_ratio"]         = feature_data["loan_amount"] / feature_data["annual_income"]
feature_data["average_monthly_transactions"] = feature_data["transaction_count"] / 6
feature_data["spending_to_income_ratio"]     = feature_data["spending_ratio"] / 100
```

💡 **Insight:** Box-Cox (needs strictly positive values) and Yeo-Johnson (handles zero/negative too) both pulled skewed financial columns closer to normal. The `ColumnTransformer` pipeline produced a reusable **1008 × 26** processed array in one step. Three new ratios — `debt_to_income_ratio`, `average_monthly_transactions`, `spending_to_income_ratio` — are more directly tied to credit risk than the raw columns.

---

## 🚀 Part H: Final Deliverable



<img width="1100" height="220" alt="pic" src="https://github.com/user-attachments/assets/c562bf5b-b958-4cf8-b02a-7c27b720074c" />


```python
final_data = data.copy()
# Median-impute numerics, mode-impute categoricals
# Extract join_year/month/day/weekday
# Add debt_to_income_ratio, average_monthly_transactions, spending_to_income_ratio
# One-hot encode gender, region, employment_type, loan_purpose
final_data.to_csv("final_customer_credit_risk_dataset.csv", index=False)
```

| Metric | Before | After |
|---|---|---|
| Rows | 1008 | 1008 |
| Missing Values | ~300+ | 0 |
| Outliers (Z-score) | 26 | Capped (Winsorized) |
| Final Columns | 18 | 36 |

💡 **Insight:** Final dataset is **1008 rows × 36 columns with zero missing values** — combining median/mode imputation, date-feature extraction, 3 engineered ratios, and one-hot encoding into a single reproducible script, ready to be split into train/test sets for a `default_flag` classification model.

---

## 📂 Project Workflow
1. **Data Acquisition** → Import CSV, JSON, SQL, and API sources
2. **Merging** → Join Transactions + Metadata + Repayment History + Economic Data
3. **Cleaning** → Compare Mean/Median/Mode, Random Sample, KNN, MICE, and Complete Case Analysis
4. **Outlier Handling** → Compare Z-score, IQR, Percentile detection + Winsorization treatment
5. **Feature Engineering** → Date extraction, Ordinal/Label/One-Hot encoding, binning (equal-width/quantile/K-Means)
6. **Feature Scaling** → Compare Standard, Normalizer, MinMax, MaxAbs, Robust scalers
7. **Feature Construction** → Log/Box-Cox/Yeo-Johnson transforms + 3 engineered risk ratios
8. **Final Deliverable** → Export `final_customer_credit_risk_dataset.csv`

## 📈 Results & Insights
- ✅ Merged 4 sources into **1008 × 18**, duplicate repayment records correctly identified (not silently dropped)
- ✅ **0 missing values** in the final dataset — KNN/MICE chosen over Complete Case Analysis (which cost ~26% of rows)
- ✅ Outliers compared across 3 methods (26 / 112 / 51) — **Winsorization** applied, all 1008 rows preserved
- ✅ Mixed variable types handled: ordinal, label, and one-hot encoding chosen per column's nature
- ✅ **RobustScaler** selected after comparing 5 scalers, best suited for outlier-heavy financial columns
- ✅ **3 new engineered features** turning raw numbers into direct credit-risk signals
- ✅ Final dataset exported as `final_customer_credit_risk_dataset.csv` — **1008 × 36**, fully ML-ready

## 📌 Expected Outcomes
- A fully merged, cleaned, and imputed customer credit dataset with zero missing values
- Outlier-controlled `annual_income`, `loan_amount`, and `credit_score` via Z-score/IQR/Percentile detection + Winsorization
- A rich, ML-ready feature set (encoded, scaled, and engineered) for credit-default risk classification
- Documented comparisons across every stage to justify each final method choice

## 🚀 Suggested Next Steps
- Train a **credit-default classification model** using `final_customer_credit_risk_dataset.csv`, with `default_flag` as the target
- Explore `debt_to_income_ratio` and `credit_score` as top predictive features for a baseline model (Logistic Regression or Random Forest)
- Use `income_group` / `high_credit_flag` to segment customers into risk tiers for business reporting

## ⚙️ Installation & Setup
```bash
git clone https://github.com/yourusername/customer-credit-risk-data-prep.git
cd customer-credit-risk-data-prep
pip install -r requirements.txt
```

## 🙏 Thank You
Thanks for checking out this project! Feedback, suggestions, and contributions are always welcome.

⭐ If you found this project helpful, don't forget to star the repository and share it.
