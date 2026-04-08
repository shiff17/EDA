import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Shiffie's Analytics <3", layout="wide")

st.title("🍒 Shiffie's Analytics <3")
st.caption("Clustering + Classification + Regression + Stability Feature Selection")

# -------------------- PREPROCESSING --------------------
def preprocess_data(df):
    df = df.copy()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Separate columns
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=np.number).columns

    # Encode categorical
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Normalize numeric
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, cat_cols, num_cols


# -------------------- CLUSTERING --------------------
def optimal_clusters(X):
    scores = []
    K = range(2, 6)
    for k in K:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        scores.append(silhouette_score(X, labels))
    return K[np.argmax(scores)]


def perform_clustering(df):
    k = optimal_clusters(df)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(df)
    return df, kmeans


def plot_clusters(df):
    pca = PCA(n_components=2)
    comp = pca.fit_transform(df.drop("cluster", axis=1))

    fig, ax = plt.subplots()
    scatter = ax.scatter(comp[:,0], comp[:,1], c=df["cluster"], cmap='viridis')
    plt.title("Cluster Visualization (PCA)")
    st.pyplot(fig)


# -------------------- STABILITY FEATURE SELECTION --------------------
def stability_feature_selection(df, target, iterations=10):
    features = df.drop(target, axis=1).columns
    scores = pd.Series(0, index=features)

    for _ in range(iterations):
        sample = df.sample(frac=0.8, replace=True)

        X = sample.drop(target, axis=1)
        y = sample[target]

        model = RandomForestClassifier()
        model.fit(X, y)

        importance = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importance.nlargest(5).index

        scores[top_features] += 1

    stability = scores / iterations
    return stability.sort_values(ascending=False)


# -------------------- CLASSIFICATION --------------------
def classification_model(df, target, features):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "cm": confusion_matrix(y_test, pred)
        }

    return results


# -------------------- REGRESSION --------------------
def regression_model(df, target, features):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred)
        }

    return results


# -------------------- INSIGHTS --------------------
def generate_insights(df, col):
    counts = df[col].value_counts()
    top = counts.idxmax()
    bottom = counts.idxmin()

    st.success(f"Highest count category: {top} ({counts.max()})")
    st.info(f"Lowest count category: {bottom} ({counts.min()})")


# -------------------- UI --------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    try:
        df_processed, cat_cols, num_cols = preprocess_data(df)

        # ---------------- CATEGORICAL ----------------
        st.header("📊 Categorical Analysis")

        col = st.selectbox("Select categorical column", cat_cols)

        fig, ax = plt.subplots()
        sns.countplot(x=df[col], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        generate_insights(df, col)

        # ---------------- CLUSTERING ----------------
        st.header("🧠 Clustering")

        df_clustered, _ = perform_clustering(df_processed)
        plot_clusters(df_clustered)

        # ---------------- FEATURE SELECTION ----------------
        st.header("🎯 Stability Feature Selection")

        target = "loan_status"
        stability = stability_feature_selection(df_processed, target)

        st.write(stability)

        top_features = stability.head(5).index.tolist()

        # ---------------- CLASSIFICATION ----------------
        st.header("🤖 Classification")

        class_results = classification_model(df_processed, target, top_features)

        for model, res in class_results.items():
            st.subheader(model)
            st.write(res)

        # ---------------- REGRESSION ----------------
        st.header("📈 Regression")

        reg_target = "loan_int_rate"
        reg_results = regression_model(df_processed, reg_target, top_features)

        for model, res in reg_results.items():
            st.subheader(model)
            st.write(res)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
