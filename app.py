# 🔥 SMART ANALYTICS SUMMARY (REAL INSIGHTS)

st.markdown("## 🎯 COMPLETE ANALYTICS SUMMARY")

insights_text = ""

# ---- 1️⃣ Loan Intent vs Loan Amount (Business Question) ----
if "loan_intent" in df.columns and "loan_amnt" in df.columns:
    
    loan_analysis = (
        df.groupby("loan_intent")["loan_amnt"]
        .mean()
        .sort_values(ascending=False)
    )

    top_intent = loan_analysis.index[0]
    top_value = loan_analysis.iloc[0]
    second_value = loan_analysis.iloc[1]
    difference = top_value - second_value

    insights_text += f"""
### 📌 Loan Behavior Insight

- **Highest average loan amount:** {top_intent} (${top_value:,.0f})
- This is **${difference:,.0f} higher** than the next category.

💡 This suggests borrowers taking loans for **{top_intent}** typically require larger funding amounts.
"""

# ---- 2️⃣ Strongest Correlation ----
numeric_df = df.select_dtypes(include=np.number)

if len(numeric_df.columns) > 1:
    
    corr_matrix = numeric_df.corr().abs()

    # Remove self-correlation
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    strongest_pair = upper.unstack().dropna().sort_values(ascending=False).index[0]
    strongest_value = upper.unstack().dropna().sort_values(ascending=False).iloc[0]

    insights_text += f"""

### 🔗 Strongest Data Relationship

- **{strongest_pair[0]} ↔ {strongest_pair[1]}**
- Correlation strength: **{strongest_value:.2f}**

💡 This means as **{strongest_pair[0]} increases**, {strongest_pair[1]} tends to change significantly as well.
This relationship is one of the strongest drivers in your dataset.
"""

# ---- 3️⃣ Feature Importance from Random Forest (if trained) ----
if 'model' in locals() and hasattr(model, "feature_importances_"):
    
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(3)

    insights_text += f"""

### 🎯 What Actually Drives Predictions

Top 3 Important Features:
- {top_features.index[0]}
- {top_features.index[1]}
- {top_features.index[2]}

💡 These variables have the strongest influence on your prediction model.
They matter more than other features when determining outcomes.
"""

# ---- 4️⃣ Dataset Health Meaning ----
dataset_status = "ANALYSIS READY" if missing_pct < 5 else "NEEDS CLEANING"

insights_text += f"""

### 📊 Data Readiness

- Missing data: {missing_pct:.1f}%

💡 With only {missing_pct:.1f}% missing values, your dataset is **{dataset_status}**
and reliable for modeling and business insights.
"""

st.markdown(insights_text)
