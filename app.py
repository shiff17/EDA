# ===============================
# 🎯 COMPLETE ANALYTICS SUMMARY
# ===============================

st.markdown("## 🎯 COMPLETE ANALYTICS SUMMARY")

summary_sections = []

# -------------------------------
# 1️⃣ Business Question Example
# -------------------------------
if "loan_intent" in df.columns and "loan_amnt" in df.columns:

    intent_analysis = (
        df.groupby("loan_intent")["loan_amnt"]
        .mean()
        .sort_values(ascending=False)
    )

    top_intent = intent_analysis.index[0]
    top_value = intent_analysis.iloc[0]

    if len(intent_analysis) > 1:
        second_value = intent_analysis.iloc[1]
        diff = top_value - second_value
    else:
        diff = 0

    summary_sections.append(f"""
### 📌 Loan Intent Analysis

• Highest average loan amount: **{top_intent}** (${top_value:,.0f})  
• This is **${diff:,.0f} higher** than the next category.

Meaning:
Borrowers selecting **{top_intent}** typically require larger funding.
This category represents higher financial demand compared to others.
""")

# -------------------------------
# 2️⃣ Strongest Correlation
# -------------------------------
numeric_df = df.select_dtypes(include=np.number)

if len(numeric_df.columns) >= 2:

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    strongest = upper.unstack().dropna().sort_values(ascending=False)

    if len(strongest) > 0:
        pair = strongest.index[0]
        corr_value = strongest.iloc[0]

        summary_sections.append(f"""
### 🔗 Strongest Relationship

• **{pair[0]} ↔ {pair[1]}**  
• Correlation strength: **{corr_value:.2f}**

Meaning:
Changes in **{pair[0]}** strongly influence **{pair[1]}**.
This relationship is one of the most important structural patterns in the dataset.
""")

# -------------------------------
# 3️⃣ Feature Importance (if ML ran)
# -------------------------------
if 'model' in locals() and hasattr(model, "feature_importances_"):

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    top3 = importances.head(3)

    summary_sections.append(f"""
### 🎯 What Drives Predictions

Top 3 Influential Features:
• {top3.index[0]}  
• {top3.index[1]}  
• {top3.index[2]}

Meaning:
These variables contribute the most to prediction accuracy.
They matter more than other features in determining outcomes.
""")

# -------------------------------
# 4️⃣ Dataset Readiness
# -------------------------------
missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
status = "ANALYSIS READY" if missing_pct < 5 else "NEEDS CLEANING"

summary_sections.append(f"""
### 📊 Dataset Health

• Missing data: {missing_pct:.1f}%  

Meaning:
With only {missing_pct:.1f}% missing values, your dataset is **{status}**
and suitable for reliable modeling and decision-making.
""")

# -------------------------------
# Display All Sections
# -------------------------------
for section in summary_sections:
    st.markdown(section)
