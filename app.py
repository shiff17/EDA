import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.impute import SimpleImputer

st.set_page_config(
    page_title="Shiffie's Dashboard 🍒",
    layout="wide",
    page_icon="🍒"
)

st.markdown("""
<style>
.main-header { text-align: center; font-size: 3rem; color: #ff1493; margin-bottom: 0rem; }
.subtitle { text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
.overview-box { 
    background: linear-gradient(135deg, #fff3f3 0%, #ffe6f2 100%);
    padding: 2rem; border-radius: 15px; border-left: 6px solid #ff1493;
    box-shadow: 0 4px 12px rgba(255,20,147,0.1); margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🍒 Shiffie\'s Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Complete Analytics • Zero Crashes • Action Plan</p>', unsafe_allow_html=True)


@st.cache_data
def safe_preprocess(df, target_col):
    try:
        df_work = df.copy()

        if target_col not in df_work.columns:
            return pd.DataFrame(), pd.Series(dtype=float)

        y = df_work[target_col]
        X = df_work.drop(columns=[target_col])

        # Impute missing values
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Encode categoricals
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]

        # Force numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Scale
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        y = pd.to_numeric(y, errors='coerce').fillna(0)

        return X, y

    except Exception:
        return pd.DataFrame(), pd.Series(dtype=float)


# Sidebar
with st.sidebar:
    st.header("🗺️ What You'll Get")
    st.markdown("""
**1️⃣** Data shapes & outliers  
**2️⃣** Predictor relationships  
**3️⃣** Business metrics  
**4️⃣** ML prediction power  
**5️⃣** Data fixes  
**📦** Complete summary!
""")


uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # --- FIXED METRICS SECTION ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{len(df):,}")

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing:,}")

    st.markdown("---")

    # Data Preview
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Target Selection
    target = st.selectbox("🎯 Select Target Column for ML", df.columns)

    if target:
        X, y = safe_preprocess(df, target)

        if not X.empty:

            st.subheader("🤖 Model Performance")

            # Decide classification vs regression
            if y.nunique() <= 10:
                model = RandomForestClassifier()
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = accuracy_score(y_test, preds)

                st.success(f"🎯 Accuracy: {score:.3f}")

            else:
                model = RandomForestRegressor()
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = r2_score(y_test, preds)

                st.success(f"📈 R² Score: {score:.3f}")

        else:
            st.error("Preprocessing failed. Please check your dataset.")
