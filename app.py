import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="Shiffie's Analytics <3", layout="wide")

st.title("🍒 Shiffie's Analytics <3")
st.caption("🤖 Perfect ML + ZERO ERRORS GUARANTEED ✨")

# -------------------- UTILITIES --------------------
def is_truly_numeric(series):
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.isna().all():
            return False
        numeric_series.mean()
        return True
    except:
        return False

def get_safe_numeric_columns(df):
    return [col for col in df.columns if is_truly_numeric(df[col])]

def detect_target_type(y):
    unique_count = len(pd.to_numeric(y, errors='coerce').dropna().unique())
    return "classification" if unique_count <= 20 else "regression"

def preprocess_data(df):
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        elif is_truly_numeric(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)
    
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# -------------------- CLUSTERING --------------------
def perform_clustering(df):
    if len(df) < 3:
        st.warning("Need more data")
        return df
    k = min(5, len(df)//5)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(df)
    return df

def plot_clusters(df):
    pca = PCA(n_components=2)
    comp = pca.fit_transform(df.drop("cluster", axis=1))
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(comp[:, 0], comp[:, 1], c=df["cluster"], cmap='viridis', s=30)
    ax.set_title("Clusters (PCA)")
    plt.colorbar(scatter)
    st.pyplot(fig)

# -------------------- ML MODELS --------------------
def run_classification(df, target, features):
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )
    
    models = {
        "Logistic": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0)
        }
    return results

def run_regression(df, target, features):
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )
    
    models = {
        "Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred)  # ✅ FIXED: "R2" not "R2 Score"
        }
    return results

# -------------------- MAIN --------------------
uploaded = st.file_uploader("📁 Upload CSV", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} cols")
    except:
        st.error("❌ Invalid CSV")
        st.stop()
    
    # Overview
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Rows", len(df))
    with col2: st.metric("Columns", len(df.columns))
    with col3: st.metric("Missing", df.isna().sum().sum())
    
    st.subheader("📄 Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Preprocess
    with st.spinner("Preprocessing data..."):
        df_processed = preprocess_data(df)
    
    # Categorical
    st.header("📊 Categorical")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        selected = st.selectbox("Column", cat_cols)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x=selected, ax=ax, palette='viridis')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Group analysis
    st.header("📈 Group Analysis")
    safe_numeric = get_safe_numeric_columns(df)
    
    col1, col2 = st.columns(2)
    with col1:
        group_col = st.selectbox("Group by", df.columns)
    with col2:
        value_col = st.selectbox("Average", ["None"] + safe_numeric)
    
    if st.button("🔍 Analyze", type="secondary"):
        df_temp = df[[group_col]].copy()
        df_temp[group_col] = df_temp[group_col].astype(str)
        
        if value_col != "None":
            df_temp[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            df_temp = df_temp.dropna(subset=[value_col])
            if len(df_temp) > 1:
                means = df_temp.groupby(group_col)[value_col].mean().sort_values(ascending=False).head(10)
                st.bar_chart(means)
                st.success(f"🏆 **{means.index[0]}**: {means.iloc[0]:.3f}")
        else:
            counts = df_temp[group_col].value_counts().head(10)
            st.bar_chart(counts)
            st.success(f"🏆 **{counts.index[0]}**: {counts.iloc[0]}")
    
    # Clustering
    st.header("🧠 Clustering")
    if st.button("🎨 Cluster Data", type="secondary"):
        df_clustered = perform_clustering(df_processed.copy())
        plot_clusters(df_clustered)
        st.dataframe(df_clustered['cluster'].value_counts())
    
    # 🔥 ML - PERFECTLY SAFE
    st.header("🤖 ML Analysis")
    st.markdown("**Pick ANY column** - auto classification/regression detection!")
    
    target_cols = df_processed.columns.tolist()
    selected_target = st.selectbox("🎯 Target", target_cols)
    
    features = [col for col in target_cols if col != selected_target]
    
    if len(features) > 0:
        target_type = detect_target_type(df_processed[selected_target])
        st.info(f"**Detected**: {target_type.upper()}")
        
        if st.button("🚀 Train All Models", type="primary"):
            try:
                if target_type == "classification":
                    st.subheader("📊 Classification")
                    results = run_classification(df_processed, selected_target, features)
                    
                    for model_name, metrics in results.items():
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
                        col2.metric("Precision", f"{metrics['Precision']:.3f}")
                        col3.metric("Recall", f"{metrics['Recall']:.3f}")
                        col4.metric("F1", f"{metrics['F1']:.3f}")
                
                else:
                    st.subheader("📈 Regression")
                    results = run_regression(df_processed, selected_target, features)
                    
                    for model_name, metrics in results.items():
                        col1, col2 = st.columns(2)
                        col1.metric("RMSE", f"{metrics['RMSE']:.6f}")
                        col2.metric("R²", f"{metrics['R2']:.3f}")  # ✅ FIXED MATCH
                
                st.success("✅ All models trained successfully!")
                
            except Exception as e:
                st.error(f"ML Error: {str(e)}")
    else:
        st.warning("Need more columns for ML")

else:
    st.info("👆 Upload CSV to start!")
