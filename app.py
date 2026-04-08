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
.overview-box { 
    background: linear-gradient(135deg, #fff3f3 0%, #ffe6f2 100%);
    padding: 2rem; 
    border-radius: 15px; 
    border-left: 6px solid #ff1493;
    box-shadow: 0 4px 12px rgba(255,20,147,0.1);
    margin: 2rem 0;
}
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

        y = df_work[target_col].copy()
        X = df_work.drop(columns=[target_col])

        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        for col in X_imputed.columns:
            if X_imputed[col].dtype == 'object':
                try:
                    le = LabelEncoder()
                    X_imputed[col] = le.fit_transform(X_imputed[col].astype(str))
                except:
                    X_imputed[col] = 0

        for col in X_imputed.columns:
            X_imputed[col] = pd.to_numeric(X_imputed[col], errors='coerce').fillna(0)

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )

        y = pd.to_numeric(y, errors='coerce').fillna(
            y.median() if y.dtype in ['float64', 'int64'] else 0
        )

        return X_scaled, y

    except Exception:
        return pd.DataFrame(), pd.Series()


# Sidebar
with st.sidebar:
    st.header("🗺️ What Each Section Does")
    st.markdown("""
**1️⃣ Distributions**  
See data shape → Spot outliers  

**2️⃣ Correlations**  
Variable links → Find predictors  

**3️⃣ Group Insights**  
Business metrics → Actionable findings  

**4️⃣ ML Prediction**  
Forecasting power → Deploy models  

**5️⃣ Data Quality**  
Fix issues → Clean foundation  

**📦 Overview** → How it ALL connects!
""")


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

    # 1️⃣ Distributions
    st.header("1️⃣ 📈 Distributions")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        col1, col2 = st.columns(2)

        with col1:
            num_col = st.selectbox("📊 Select", numeric_cols)
            fig = px.histogram(df, x=num_col, marginal="box")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Mean", f"{df[num_col].mean():.1f}")

    # 2️⃣ Correlations
    st.header("2️⃣ 🔗 Correlations")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

    # 3️⃣ Group Insights
    st.header("3️⃣ 📊 Group Insights")

    if cat_cols and numeric_cols:
        col1, col2 = st.columns(2)

        with col1:
            group_col = st.selectbox("Group", cat_cols)

        with col2:
            value_col = st.selectbox("Average", numeric_cols)

        if st.button("🔍 Analyze"):
            result = df.groupby(group_col)[value_col].mean().sort_values(ascending=False)
            fig = px.bar(x=result.index, y=result.values, text=result.values.round(1))
            st.plotly_chart(fig, use_container_width=True)

    # 4️⃣ ML Prediction
    st.header("4️⃣ 🎯 ML Prediction")

    target_col = st.selectbox("🎯 Predict", df.columns)

    if st.button("🚀 Train Models", type="primary"):

        X, y = safe_preprocess(df, target_col)

        if len(X) > 10 and len(y) > 10:

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            unique_y = len(pd.unique(y_train))
            is_class = unique_y <= 20

            models = {
                "Linear/Logistic": LinearRegression() if not is_class else LogisticRegression(max_iter=1000),
                "🌳 Random Forest": RandomForestRegressor(n_estimators=50) if not is_class else RandomForestClassifier(n_estimators=50)
            }

            results = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                score = r2_score(y_test, preds) if not is_class else accuracy_score(y_test, preds)
                results[name] = score

            st.subheader("📊 Results")

            result_df = pd.DataFrame(list(results.items()), columns=['Model', 'Score']).round(3)
            st.dataframe(result_df.style.highlight_max(axis=0))

            best_row = result_df.loc[result_df['Score'].idxmax()]
            best_model_name = best_row['Model']
            best_score = best_row['Score']

            if best_score >= 0.8:
                st.success(f"🚀 EXCELLENT: {best_score:.3f}")
            elif best_score >= 0.5:
                st.info(f"✅ GOOD: {best_score:.3f}")
            else:
                st.warning(f"⚠️ WEAK: {best_score:.3f}")

        else:
            st.warning("Not enough clean data for ML")

    # 5️⃣ Data Quality
    st.header("5️⃣ 🛠️ Data Quality")

    missing = df.isna().sum()

    if missing.any():
        missing_df = pd.DataFrame({
            'Count': missing,
            '%': (missing/len(df)*100).round(1)
        }).sort_values('%', ascending=False)

        st.dataframe(missing_df[missing_df['Count'] > 0])

    st.markdown("---")

    # 🔥 FINAL SUMMARY BOX (FIXED)

    top_group = "Run group analysis"
    ml_score_text = "Run ML section"
    best_model_display = "Run ML section"

    if 'result_df' in locals():
        best_row = result_df.loc[result_df['Score'].idxmax()]
        best_model_display = best_row['Model']
        ml_score_text = f"{best_row['Score']:.3f}"

    dataset_status = "ANALYSIS READY" if missing_pct < 5 else "PARTIALLY READY"
    clean_status = "✅ READY" if missing_pct < 5 else "⚠️ CLEAN NEEDED"

    st.markdown(f"""
    <div class="overview-box">
    <h2 style='color: #ff1493;'>🎯 COMPLETE ANALYTICS SUMMARY</h2>

    <h3>📊 Dataset Status</h3>
    - <b>Rows</b>: {len(df):,} | <b>Columns</b>: {len(df.columns)} | <b>Missing</b>: {missing_pct:.1f}%  
    {clean_status}

    <h3>🔗 Key Findings</h3>
    - <b>Strongest correlation</b>: Check heatmap  
    - <b>Prediction power</b>: {ml_score_text}

    <h3>🚀 Action Plan</h3>
    1. Clean {missing_pct:.1f}% missing data  
    2. Focus on top correlated features  
    3. Deploy best model ({best_model_display})  
    4. Monitor business metrics  

    <h3 style='color: #ff1493;'>💡 YOUR DATASET IS {dataset_status}!</h3>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<p style='text-align: center; color: #ff1493; font-size: 1.1rem; margin-top: 2rem;'>🍒 Shiffie's Dashboard | Analytics Mastered ✨</p>", unsafe_allow_html=True)
