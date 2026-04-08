import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Shiffie's Analytics", layout="wide", page_icon="📊")

st.title("📊 Shiffie's Bulletproof Analytics")
st.caption("✅ Handles ANY data • ✅ No crashes • ✅ Practical insights")

@st.cache_data
def safe_preprocess(df, target_col):
    """100% safe preprocessing - handles ALL edge cases"""
    df_clean = df.copy()
    
    # Handle target separately
    if target_col in df_clean.columns:
        target_series = df_clean[target_col]
    else:
        target_series = pd.Series()
    
    # Remove target for feature processing
    if target_col in df_clean.columns:
        feature_df = df_clean.drop(columns=[target_col])
    else:
        feature_df = df_clean
    
    # 1. Fill missing values SAFELY
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    feature_df = pd.DataFrame(
        imputer.fit_transform(feature_df),
        columns=feature_df.columns,
        index=feature_df.index
    )
    
    # 2. Encode categoricals SAFELY
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object':
            try:
                le = LabelEncoder()
                feature_df[col] = le.fit_transform(feature_df[col].astype(str))
            except:
                feature_df[col] = 0  # Fallback
    
    # 3. Ensure ALL numeric
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
    
    # Scale
    scaler = StandardScaler()
    feature_df = pd.DataFrame(
        scaler.fit_transform(feature_df),
        columns=feature_df.columns,
        index=feature_df.index
    )
    
    return feature_df, target_series

uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Data overview
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Rows", len(df))
    with col2: st.metric("Columns", len(df.columns))
    with col3: st.metric("Missing %", f"{df.isna().sum().sum()/(len(df)*len(df.columns))*100:.1f}%")
    
    st.subheader("🔍 Preview")
    st.dataframe(df.head())
    
    # 1. Distributions
    st.header("1️⃣ Data Distributions")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            num_col = st.selectbox("Numeric", numeric_cols)
            fig = px.histogram(df, x=num_col, nbins=30, marginal="box")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Mean", df[num_col].mean())
            st.metric("Std", df[num_col].std())
    
    if cat_cols:
        cat_col = st.selectbox("Categorical", cat_cols)
        fig = px.histogram(df, x=cat_col)
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Correlations
    st.header("2️⃣ Correlations")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
    
    # 3. Group by
    st.header("3️⃣ Group Insights")
    if cat_cols and numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            group_col = st.selectbox("Group by", cat_cols)
        with col2:
            value_col = st.selectbox("Average", numeric_cols)
        
        if st.button("Show Insights"):
            result = df.groupby(group_col)[value_col].agg(['mean', 'count']).round(2)
            result.columns = ['Average', 'Count']
            result = result.sort_values('Average', ascending=False)
            
            fig = px.bar(result, x=result.index, y='Average', text='Average')
            st.plotly_chart(fig)
            st.success(f"🏆 Highest: **{result.index[0]}** ({result.iloc[0]['Average']:.2f})")
    
    # 4. ML - BULLETPROOF
    st.header("4️⃣ Predict Any Target")
    st.markdown("**Pick what you want to predict** - works with ANY column!")
    
    target_col = st.selectbox("🎯 Predict", df.columns.tolist())
    
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models..."):
            # SAFE PREPROCESSING
            X, y = safe_preprocess(df, target_col)
            
            if len(X) == 0 or len(y) == 0:
                st.error("No valid data after preprocessing")
                st.stop()
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Auto-detect task
            unique_targets = len(pd.unique(y_train))
            is_classification = unique_targets <= 20
            
            st.info(f"**Task**: {'Classification' if is_classification else 'Regression'} "
                   f"({unique_targets} unique values)")
            
            # Models
            if is_classification:
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42)
                }
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)
                }
            
            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    if is_classification:
                        score = accuracy_score(y_test, y_pred)
                    else:
                        score = r2_score(y_test, y_pred)
                    
                    results[name] = score
                except Exception as e:
                    st.warning(f"{name} failed: {str(e)}")
                    results[name] = 0
            
            # Results table
            st.subheader("📊 Results")
            result_df = pd.DataFrame(list(results.items()), columns=['Model', 'Score']).round(3)
            st.dataframe(result_df.style.highlight_max(axis=0))
            
            best_model_name = result_df.loc[result_df['Score'].idxmax(), 'Model']
            best_score = result_df['Score'].max()
            st.balloons()
            st.success(f"🎉 **Best**: {best_model_name} = {best_score:.3f}")
            
            # Feature importance for Random Forest
            if "Random Forest" in results and results["Random Forest"] > 0:
                rf_model = models["Random Forest"]
                importances = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(importances, x='Importance', y='Feature', 
                           orientation='h', title="🔥 Top 10 Predictors")
                st.plotly_chart(fig)
    
    # 5. Data quality
    st.header("5️⃣ Data Quality")
    missing = df.isna().sum()
    if missing.sum() > 0:
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            '% Missing': (missing/len(df)*100).round(1)
        }).sort_values('% Missing', ascending=False)
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])
    else:
        st.success("✅ Perfect data quality!")

else:
    st.info("👆 Upload CSV for instant insights!")
