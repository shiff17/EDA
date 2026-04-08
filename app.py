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
    padding: 2rem; border-radius: 15px; border-left: 6px solid #ff1493;
    box-shadow: 0 4px 12px rgba(255,20,147,0.1); margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🍒 Shiffie\'s Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Complete Analytics • Zero Crashes • Action Plan</p>', unsafe_allow_html=True)

@st.cache_data
def safe_preprocess(df, target_col):
    """Bulletproof preprocessing"""
    try:
        df_work = df.copy()
        if target_col in df_work.columns:
            y = df_work[target_col]
            X = df_work.drop(columns=[target_col])
        else:
            return pd.DataFrame(), pd.Series(dtype=float)
        
        # Impute missing
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Encode categoricals
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(pd.factorize(X[col])[0], errors='coerce').fillna(0)
        
        # Force numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Scale
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        y = pd.to_numeric(y, errors='coerce').fillna(0)
        
        return X, y
    except:
        return pd.DataFrame(), pd.Series(dtype=float)

# Sidebar
with st.sidebar:
    st.header("🗺️ What You'll Get")
    st.markdown("**1️⃣** Data shapes & outliers\n**2️⃣** Predictor relationships\n**3️⃣** Business metrics\n**4️⃣** ML prediction power\n**5️⃣** Data fixes\n**📦** Complete summary!")

uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Metrics
    col1, col2, col3
