import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Shiffie's Analytics", layout="wide", page_icon="📊")

st.title("📊 Shiffie's Smart Analytics")
st.caption("✅ Practical insights • ✅ No crashes • ✅ Clear explanations")

# File upload
uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    with col4:
        st.metric("Unique Values", df.nunique().mean())
    
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # 1. SIMPLE EDA - Most useful first
    st.header("1️⃣ Quick EDA - Find Patterns Instantly")
    st.markdown("""
    **What this does**: Shows distributions and relationships in your data
    **Why useful**: Spot outliers, trends, missing values immediately
    """)
    
    # Auto-detect column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Numeric distributions
    if numeric_cols:
        st.subheader("📈 Numeric Columns Distribution")
        selected_num = st.selectbox("Pick numeric column", numeric_cols)
        fig = px.histogram(df, x=selected_num, marginal="box", title=f"Distribution of {selected_num}")
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean", f"{df[selected_num].mean():.2f}")
        with col2:
            st.metric("Missing", f"{df[selected_num].isna().sum()}")
    
    # Categorical distributions
    if categorical_cols:
        st.subheader("📊 Categorical Columns")
        selected_cat = st.selectbox("Pick categorical column", categorical_cols)
        fig = px.histogram(df, x=selected_cat, title=f"Count of {selected_cat}")
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. CORRELATION - SUPER USEFUL
    st.header("2️⃣ Correlations - Find Relationships")
    st.markdown("""
    **What this shows**: Which columns move together
    **Why useful**: Find predictors for your target variable
    """)
    
    if len(numeric_cols) >= 2:
        corr_data = df[numeric_cols].corr()
        fig = px.imshow(corr_data, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        fig.update_layout(title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 2+ numeric columns for correlation")
    
    # 3. GROUP ANALYSIS - Business Insights
    st.header("3️⃣ Group Analysis - Business Insights")
    st.markdown("""
    **What this does**: Average values by category
    **Why useful**: "What's the average loan amount by home ownership?"
    """)
    
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        col1, col2 = st.columns(2)
        with col1:
            group_by = st.selectbox("Group by (category)", categorical_cols)
        with col2:
            avg_col = st.selectbox("Average (numeric)", numeric_cols)
        
        if st.button("🔍 Show Insights"):
            result = df.groupby(group_by)[avg_col].agg(['mean', 'count']).round(2)
            result.columns = ['Average', 'Count']
            result = result.sort_values('Average', ascending=False)
            
            fig = px.bar(result, x=result.index, y='Average', 
                        text='Average', title=f"Average {avg_col} by {group_by}")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(result)
            st.success(f"🏆 Highest: **{result.index[0]}** = {result.iloc[0]['Average']:.2f}")
    
    # 4. ML PREDICTION - Only if target selected
    st.header("4️⃣ Predict Target Variable")
    st.markdown("""
    **What this does**: Predicts your target using all other columns
    **Why useful**: "Can I predict loan approval from other features?"
    """)
    
    all_cols = df.columns.tolist()
    target_col = st.selectbox("🎯 What do you want to predict?", all_cols)
    
    if st.button("🚀 Train Predictor", type="primary"):
        # Prepare data
        feature_cols = [col for col in all_cols if col != target_col]
        
        # Encode categoricals
        df_ml = df[feature_cols + [target_col]].copy()
        for col in df_ml.columns:
            if df_ml[col].dtype == 'object':
                df_ml[col] = LabelEncoder().fit_transform(df_ml[col].astype(str))
        
        # Remove any remaining non-numeric
        df_ml = df_ml.select_dtypes(include=[np.number])
        if target_col not in df_ml.columns:
            st.error("Target column became non-numeric - pick another")
            st.stop()
        
        feature_cols = [col for col in df_ml.columns if col != target_col]
        
        if len(feature_cols) == 0:
            st.error("No features available")
            st.stop()
        
        X = df_ml[feature_cols]
        y = df_ml[target_col]
        
        # Auto-detect task type
        unique_y = len(y.unique())
        is_classification = unique_y <= 20
        
        st.info(f"**Task**: {'Classification' if is_classification else 'Regression'} "
                f"({unique_y} unique target values)")
        
        # Train models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if is_classification:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100)
            }
            metric_name = "Accuracy"
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100)
            }
            metric_name = "R² Score"
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if is_classification:
                score = accuracy_score(y_test, y_pred)
            else:
                score = r2_score(y_test, y_pred)
            
            results[name] = score
        
        # Display results
        st.subheader("📊 Model Performance")
        perf_df = pd.DataFrame({
            'Model': list(results.keys()),
            f'{metric_name}': list(results.values())
        }).round(3)
        
        st.dataframe(perf_df, use_container_width=True)
        
        best_model = max(results, key=results.get)
        st.success(f"🏆 **Best Model**: {best_model} ({results[best_model]:.3f})")
        
        # Feature importance (if RF)
        if "Random Forest" in best_model:
            rf_model = models["Random Forest"]
            importances = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.subheader("🔥 Top Predictors")
            fig = px.bar(importances.head(10), x='Importance', y='Feature',
                        orientation='h', title="Most Important Features")
            st.plotly_chart(fig, use_container_width=True)
    
    # 5. Missing Data Report
    st.header("5️⃣ Data Quality Check")
    missing_data = df.isna().sum()
    if missing_data.sum() > 0:
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing': missing_data.values,
            '% Missing': (missing_data / len(df) * 100).round(1)
        }).sort_values('% Missing', ascending=False)
        
        st.dataframe(missing_df[missing_df['Missing'] > 0])
    else:
        st.success("✅ No missing data!")

else:
    st.info("👆 **Upload your CSV** to get instant insights!")
    st.markdown("""
    ## 🎯 **What you'll get:**
    1. **Distributions** - See patterns in your data
    2. **Correlations** - Find related variables  
    3. **Group insights** - Business metrics by category
    4. **ML predictions** - Predict any target automatically
    5. **Data quality** - Missing values report
    
    **No crashes. No complexity. Just insights.**
    """)
    
