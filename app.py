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

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Shiffie's Analytics <3", layout="wide")

st.title("🍒 Shiffie's Analytics <3")
st.caption("🤖 Smart ML + Zero Errors + Auto-Detection ✨")

# -------------------- SAFE NUMERIC CHECK --------------------
def is_truly_numeric(series):
    """Test if column can compute mean"""
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.isna().all():
            return False
        numeric_series.mean()
        return True
    except:
        return False

def get_safe_numeric_columns(df):
    """Get columns that can safely compute mean"""
    return [col for col in df.columns if is_truly_numeric(df[col])]

def detect_target_type(y):
    """Auto-detect if target is classification or regression"""
    unique_count = len(y.dropna().unique())
    y_numeric = pd.to_numeric(y, errors='coerce')
    
    # Classification: few unique values (<20) OR already categorical
    if unique_count <= 20 or not is_truly_numeric(y):
        return "classification"
    # Regression: many unique values AND numeric
    return "regression"

# -------------------- PREPROCESSING --------------------
def preprocess_data(df):
    df = df.copy()
    
    # Safe fillna
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        elif is_truly_numeric(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)
    
    # Encode categoricals
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # Final numeric conversion
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# [Keep clustering functions same as before]
def optimal_clusters(X):
    n_samples = X.shape[0]
    K = range(2, min(6, n_samples//3 + 1))
    if len(K) == 0:
        return 2
    scores = []
    for k in K:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        scores.append(silhouette_score(X, labels))
    return K[np.argmax(scores)]

def perform_clustering(df):
    k = optimal_clusters(df)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(df)
    return df

def plot_clusters(df):
    if len(df) < 2:
        st.warning("Not enough data for clustering")
        return
    pca = PCA(n_components=2)
    comp = pca.fit_transform(df.drop("cluster", axis=1))
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(comp[:, 0], comp[:, 1], c=df["cluster"], cmap='viridis', s=50)
    ax.set_title("Clusters (PCA)")
    plt.colorbar(scatter)
    st.pyplot(fig)

# -------------------- ML FUNCTIONS --------------------
def stability_feature_selection(df, target, iterations=10):
    features = df.drop(target, axis=1).columns
    scores = pd.Series(0, index=features)
    for _ in range(iterations):
        sample = df.sample(frac=0.8, replace=True)
        X = sample.drop(target, axis=1)
        y = sample[target]
        model = RandomForestClassifier(random_state=42)  # Works for both
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importance.nlargest(5).index
        scores[top_features] += 1
    return (scores / iterations).sort_values(ascending=False)

def run_classification(df, target, features):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0)
        }
    return results

def run_regression(df, target, features):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred)
        }
    return results

# -------------------- MAIN --------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("✅ File loaded!")
    
    st.subheader("📄 Preview")
    st.dataframe(df.head())
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Rows", len(df))
    with col2: st.metric("Columns", len(df.columns))
    with col3: st.metric("Nulls", df.isna().sum().sum())
    
    try:
        df_processed = preprocess_data(df)
        
        # Categorical
        st.header("📊 Categorical")
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            col = st.selectbox("Column", cat_cols)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(data=df, x=col, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Aggregated Analysis
        st.header("📈 Group Analysis")
        safe_numeric = get_safe_numeric_columns(df)
        
        col1, col2 = st.columns(2)
        with col1:
            x = st.selectbox("Group by", df.columns)
        with col2:
            y = st.selectbox("Average", ["None"] + safe_numeric)
        
        if st.button("Analyze", type="primary"):
            df_ana = df.copy()
            df_ana[x] = df_ana[x].astype(str)
            
            if y != "None":
                df_ana[y] = pd.to_numeric(df_ana[y], errors='coerce')
                df_ana = df_ana.dropna(subset=[y])
                
                if len(df_ana) > 1:
                    result = df_ana.groupby(x)[y].mean().sort_values(ascending=False).head(10)
                    st.bar_chart(result)
                    st.success(f"Top: {result.index[0]} = {result.iloc[0]:.3f}")
                else:
                    st.warning("No valid data")
            else:
                counts = df_ana[x].value_counts().head(10)
                st.bar_chart(counts)
                st.success(f"Top: {counts.index[0]} = {counts.iloc[0]}")
        
        # Clustering
        st.header("🧠 Clustering")
        if st.button("Cluster", type="secondary"):
            df_clustered = perform_clustering(df_processed.copy())
            plot_clusters(df_clustered)
        
        # 🚀 SMART ML
        st.header("🤖 SMART ML")
        ml_cols = df_processed.select_dtypes(include='number').columns.tolist()
        target = st.selectbox("Target", ml_cols)
        
        if st.button("🚀 RUN ML", type="primary"):
            # Feature selection
            stability = stability_feature_selection(df_processed, target)
            st.subheader("🔥 Top Features")
            st.dataframe(stability.head(10).round(3))
            
            top_features = stability.head(8).index.tolist()
            
            # AUTO-DETECT TARGET TYPE
            target_type = detect_target_type(df_processed[target])
            st.success(f"🎯 Auto-detected: **{target_type.upper()}** target")
            
            if target_type == "classification":
                st.subheader("📊 Classification Results")
                try:
                    results = run_classification(df_processed, target, top_features)
                    for model, metrics in results.items():
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Acc", f"{metrics['Accuracy']:.3f}")
                        col2.metric("Prec", f"{metrics['Precision']:.3f}")
                        col3.metric("Rec", f"{metrics['Recall']:.3f}")
                        col4.metric("F1", f"{metrics['F1 Score']:.3f}")
                except Exception as e:
                    st.error(f"Classification failed: {e}")
            
            else:  # regression
                st.subheader("📈 Regression Results")
                try:
                    results = run_regression(df_processed, target, top_features)
                    for model, metrics in results.items():
                        col1, col2 = st.columns(2)
                        col1.metric("RMSE", f"{metrics['RMSE']:.3f}")
                        col2.metric("R²", f"{metrics['R2 Score']:.3f}")
                except Exception as e:
                    st.error(f"Regression failed: {e}")

    except Exception as e:
        st.error(f"Error: {e}")
