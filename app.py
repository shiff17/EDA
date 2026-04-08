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

st.set_page_config(
    page_title="Shiffie's Dashboard 🍒", 
    layout="wide", 
    page_icon="🍒",
    initial_sidebar_state="expanded"
)

# Custom CSS for centering title
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-size: 3rem;
        color: #ff1493;
        margin-bottom: 0rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Centered title
st.markdown('<h1 class="main-header">🍒 Shiffie\'s Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your Complete Data Analysis Partner ✨</p>', unsafe_allow_html=True)

# Sidebar roadmap
with st.sidebar:
    st.header("🗺️ What You'll Discover")
    st.markdown("""
    **1️⃣ Distributions**  
    *See patterns, outliers, trends instantly*
    
    **2️⃣ Correlations**  
    *Find which variables predict each other*
    
    **3️⃣ Group Insights**  
    *Business metrics: "Avg income by home ownership?"*
    
    **4️⃣ ML Prediction**  
    *Predict ANY target: "Can income predict loan approval?"*
    
    **5️⃣ Data Quality**  
    *Fix missing values & data issues*
    
    **💡 Each section explains what it means for YOUR analysis**
    """)

# Main app
uploaded_file = st.file_uploader("📁 Upload CSV", type="csv", help="Drag & drop or click to upload")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Data overview
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("📊 Rows", len(df))
    with col2: st.metric("📋 Columns", len(df.columns))
    with col3: 
        missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        st.metric("🔍 Missing Data", f"{missing_pct:.1f}%")
    
    st.markdown("---")
    
    # 1. Distributions
    st.header("1️⃣ 📈 Data Distributions")
    st.markdown("""
    **🔍 What you see**: Shape of your data (normal? skewed? outliers?)
    **💡 What it means**: 
    - **Normal distribution** = typical values
    - **Skewed** = extremes dominate  
    - **Outliers** = data quality issues or interesting cases
    **✅ Action**: Spot problems before analysis
    """)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            num_col = st.selectbox("📊 Numeric column", numeric_cols, key="dist_num")
            fig = px.histogram(df, x=num_col, nbins=30, marginal="box", 
                             title=f"🍒 Distribution: {num_col}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("📈 Mean", f"{df[num_col].mean():.1f}")
            st.metric("📊 Median", f"{df[num_col].median():.1f}")
            st.metric("🚨 Outliers", f"{((df[num_col] > df[num_col].mean() + 3*df[num_col].std()).sum())}")
    
    if cat_cols:
        st.subheader("📋 Categorical")
        cat_col = st.selectbox("📋 Category", cat_cols, key="dist_cat")
        fig = px.histogram(df, x=cat_col, title=f"🍒 Counts: {cat_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 2. Correlations
    st.header("2️⃣ 🔗 Correlations")
    st.markdown("""
    **🔍 What you see**: How variables move together (red=positive, blue=negative)
    **💡 What it means**:
    - **High correlation (>0.7)** = strong relationship
    - **Target column correlations** = best predictors
    - **Perfect correlation (1.0)** = redundant columns
    **✅ Action**: Pick top correlated features for prediction
    """)
    
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                       title="🍒 Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations
        top_corr = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
        st.subheader("🏆 Top 5 Relationships")
        st.dataframe(top_corr.head(10))
    
    st.markdown("---")
    
    # 3. Group insights
    st.header("3️⃣ 📊 Business Insights")
    st.markdown("""
    **🔍 What you see**: Average values by category
    **💡 What it means**:
    - **Highest average** = best performing group
    - **Lowest average** = problem area
    - **Count** = sample size reliability
    **✅ Action**: Focus business efforts on top/bottom groups
    """)
    
    if cat_cols and numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            group_col = st.selectbox("📂 Group by", cat_cols, key="group_cat")
        with col2:
            value_col = st.selectbox("📊 Average", numeric_cols, key="group_num")
        
        if st.button("🔍 Generate Insights", type="primary"):
            result = df.groupby(group_col)[value_col].agg(['mean', 'count']).round(2)
            result.columns = ['🍒 Average', '📊 Count']
            result = result.sort_values('🍒 Average', ascending=False)
            
            fig = px.bar(result, x=result.index, y='🍒 Average', text='🍒 Average',
                        title=f"🍒 {value_col} by {group_col}")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("💡 Key Insights")
            st.markdown(f"""
            - **🏆 BEST**: {result.index[0]} ({result.iloc[0]['🍒 Average']:.2f})
            - **📉 WORST**: {result.index[-1]} ({result.iloc[-1]['🍒 Average']:.2f})
            - **⚠️ Small sample**: Groups with <10 observations
            """)
            st.dataframe(result)
    
    st.markdown("---")
    
    # 4. ML Prediction
    st.header("4️⃣ 🎯 Predict Any Target")
    st.markdown("""
    **🔍 What you see**: How well models predict your target
    **💡 What it means**:
    - **Score 0.8+** = Excellent prediction power
    - **Score 0.5-0.8** = Good, actionable insights  
    - **Score <0.5** = Hard to predict (random factors)
    **✅ Action**: Use top model + top features for predictions
    """)
    
    target_col = st.selectbox("🎯 Predict this column", df.columns.tolist())
    
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("🚀 Train All Models", type="primary", use_container_width=True):
            with st.spinner("Analyzing prediction power..."):
                # SAFE PROCESSING
                X, y = safe_preprocess(df, target_col)
                
                if len(X) == 0:
                    st.error("❌ No valid features found")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    unique_targets = len(pd.unique(y_train))
                    is_classification = unique_targets <= 20
                    
                    st.info(f"**📊 Task**: {'Classification' if is_classification else 'Regression'}")
                    
                    models = {
                        "🍒 Logistic/Linear": LogisticRegression(max_iter=1000) if is_classification else LinearRegression(),
                        "🌳 Random Forest": RandomForestClassifier(n_estimators=50) if is_classification else RandomForestRegressor(n_estimators=50)
                    }
                    
                    results = {}
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        if is_classification:
                            score = accuracy_score(y_test, y_pred)
                        else:
                            score = r2_score(y_test, y_pred)
                        
                        results[name] = score
                    
                    # Results
                    st.subheader("📈 Prediction Power")
                    result_df = pd.DataFrame(list(results.items()), columns=['Model', 'Score']).round(3)
                    st.dataframe(result_df.style.highlight_max(axis=0))
                    
                    best_score = result_df['Score'].max()
                    if best_score > 0.8:
                        st.balloons()
                        st.success(f"🎉 EXCELLENT PREDICTION POWER: {best_score:.3f}")
                    elif best_score > 0.5:
                        st.info(f"✅ GOOD PREDICTOR: {best_score:.3f}")
                    else:
                        st.warning(f"⚠️ WEAK PREDICTOR: {best_score:.3f}")
    
    st.markdown("---")
    
    # 5. Data quality
    st.header("5️⃣ 🛠️ Data Quality Report")
    st.markdown("""
    **🔍 What you see**: Missing data problems
    **💡 What it means**:
    - **>5% missing** = Clean before analysis
    - **>20% missing** = Drop column or collect more data
    **✅ Action**: Fix data quality first
    """)
    
    missing = df.isna().sum()
    if missing.sum() > 0:
        missing_df = pd.DataFrame({
            '🍒 Missing Count': missing,
            '📊 % Missing': (missing/len(df)*100).round(1)
        }).sort_values('📊 % Missing', ascending=False)
        
        st.dataframe(missing_df[missing_df['🍒 Missing Count'] > 0])
        
        critical = missing_df[missing_df['📊 % Missing'] > 5]
        if len(critical) > 0:
            st.error(f"🚨 {len(critical)} columns need cleaning!")
        else:
            st.success("✅ Data quality OK")
    else:
        st.success("🎉 Perfect data quality!")

# Safe preprocess function (same as before)
@st.cache_data
def safe_preprocess(df, target_col):
    df_clean = df.copy()
    
    if target_col in df_clean.columns:
        target_series = df_clean[target_col]
        feature_df = df_clean.drop(columns=[target_col])
    else:
        target_series = pd.Series()
        feature_df = df_clean
    
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    feature_df = pd.DataFrame(
        imputer.fit_transform(feature_df),
        columns=feature_df.columns
    )
    
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object':
            try:
                le = LabelEncoder()
                feature_df[col] = le.fit_transform(feature_df[col].astype(str))
            except:
                feature_df[col] = 0
        
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
    
    scaler = StandardScaler()
    feature_df = pd.DataFrame(
        scaler.fit_transform(feature_df),
        columns=feature_df.columns
    )
    
    return feature_df, target_series

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>🍒 Made with ❤️ by Shiffie | Bulletproof Analytics</p>", unsafe_allow_html=True)
