import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Shiffie's Dashboard 🍒", layout="wide", page_icon="🍒")

st.markdown("""
<style>
.main-header { text-align: center; font-size: 3rem; color: #ff1493; margin-bottom: 0rem; }
.subtitle { text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🍒 Shiffie\'s Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Complete Analytics • Zero Crashes • Actionable Insights</p>', unsafe_allow_html=True)


@st.cache_data
def safe_preprocess(df, target_col):
    try:
        df_work = df.copy()

        if target_col not in df_work.columns:
            return pd.DataFrame(), pd.Series()

        y = df_work[target_col]
        X = df_work.drop(columns=[target_col])

        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        y = pd.to_numeric(y, errors='coerce').fillna(0)

        return X, y

    except:
        return pd.DataFrame(), pd.Series()


uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Rows", len(df))
    with col2:
        st.metric("📋 Columns", len(df.columns))
    with col3:
        missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        st.metric("🔍 Missing", f"{missing_pct:.1f}%")

    st.markdown("---")

    # Distributions
    st.header("1️⃣ 📈 Distributions")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if numeric_cols:
        num_col = st.selectbox("Select Numeric Column", numeric_cols)
        fig = px.histogram(df, x=num_col, marginal="box")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation
    st.header("2️⃣ 🔗 Correlations")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

    # Group Analysis
    st.header("3️⃣ 📊 Group Insights")
    if cat_cols and numeric_cols:
        group_col = st.selectbox("Group By", cat_cols)
        value_col = st.selectbox("Analyze Value", numeric_cols)

        if st.button("Analyze Groups"):
            result = df.groupby(group_col)[value_col].mean().sort_values(ascending=False)
            fig = px.bar(x=result.index, y=result.values, text=result.values.round(1))
            st.plotly_chart(fig, use_container_width=True)

    # ML Section
    st.header("4️⃣ 🎯 ML Prediction")
    target_col = st.selectbox("Select Target", df.columns)

    model = None
    X = None

    if st.button("Train Models"):

        X, y = safe_preprocess(df, target_col)

        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            is_class = len(pd.unique(y_train)) <= 20

            model = (
                RandomForestClassifier(n_estimators=50)
                if is_class else
                RandomForestRegressor(n_estimators=50)
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            score = (
                accuracy_score(y_test, preds)
                if is_class else
                r2_score(y_test, preds)
            )

            st.success(f"Model Score: {score:.3f}")

    # -----------------------------
    # 🎯 SMART ANALYTICS SUMMARY
    # -----------------------------

    st.markdown("## 🎯 COMPLETE ANALYTICS SUMMARY")

    summary = []

    # Loan intent question
    if "loan_intent" in df.columns and "loan_amnt" in df.columns:
        intent_analysis = (
            df.groupby("loan_intent")["loan_amnt"]
            .mean()
            .sort_values(ascending=False)
        )

        top_intent = intent_analysis.index[0]
        top_value = intent_analysis.iloc[0]
        second_value = intent_analysis.iloc[1] if len(intent_analysis) > 1 else 0
        diff = top_value - second_value

        summary.append(f"""
### 📌 Which loan intent has the highest average loan amount?

**{top_intent}** has the highest average loan amount at ${top_value:,.0f}.  
It exceeds the next category by ${diff:,.0f}.

Meaning:
This intent category typically requires larger capital.
This may indicate higher-cost financial needs or consolidation behavior.
""")

    # Strongest correlation
    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        strongest = upper.unstack().dropna().sort_values(ascending=False)

        if len(strongest) > 0:
            pair = strongest.index[0]
            corr_value = strongest.iloc[0]

            summary.append(f"""
### 🔗 Strongest Feature Relationship

**{pair[0]} and {pair[1]}** show the strongest correlation ({corr_value:.2f}).

Meaning:
As one increases, the other changes significantly.
This relationship is structurally important in the dataset.
""")

    # Feature importance
    if model is not None and hasattr(model, "feature_importances_") and X is not None:
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top3 = importances.sort_values(ascending=False).head(3)

        summary.append(f"""
### 🎯 What Features Matter Most?

Top 3 drivers of prediction:
• {top3.index[0]}  
• {top3.index[1]}  
• {top3.index[2]}

Meaning:
These variables contribute most to model decisions.
They are key predictors influencing outcomes.
""")

    # Dataset readiness
    missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
    status = "ANALYSIS READY" if missing_pct < 5 else "NEEDS CLEANING"

    summary.append(f"""
### 📊 Dataset Readiness

Missing Data: {missing_pct:.1f}%  

Your dataset is **{status}** and suitable for business insight generation.
""")

    for section in summary:
        st.markdown(section)


st.markdown(
    "<p style='text-align:center;color:#ff1493;'>🍒 Shiffie's Dashboard | Analytics Mastered</p>",
    unsafe_allow_html=True
)
