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

    # Encode categorical
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Convert everything to numeric safely
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill any NaNs created
    df = df.fillna(0)

    # Normalize
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df


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
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(df)
    return df


def plot_clusters(df):
    pca = PCA(n_components=2)
    comp = pca.fit_transform(df.drop("cluster", axis=1))

    fig, ax = plt.subplots()
    ax.scatter(comp[:, 0], comp[:, 1], c=df["cluster"])
    ax.set_title("Cluster Visualization (PCA)")
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
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test, pred)
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
            "R2 Score": r2_score(y_test, pred)
        }

    return results


# -------------------- INSIGHTS --------------------
def generate_insights(df, col):
    counts = df[col].astype(str).value_counts()

    top = counts.idxmax()
    bottom = counts.idxmin()

    st.success(f"Highest count category: {top} ({counts.max()})")
    st.info(f"Lowest count category: {bottom} ({counts.min()})")

    st.caption("This represents frequency (count), not averages.")


# -------------------- FILE UPLOAD --------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    try:
        df_processed = preprocess_data(df)

        # ---------------- CATEGORICAL ----------------
        st.header("📊 Categorical Analysis")

        cat_cols = df.select_dtypes(include='object').columns.tolist()

        if len(cat_cols) == 0:
            st.warning("No categorical columns found.")
        else:
            col = st.selectbox("Select categorical column", cat_cols)

            fig, ax = plt.subplots()
            sns.countplot(x=df[col].astype(str), ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            generate_insights(df, col)

        # ---------------- CLUSTERING ----------------
        st.header("🧠 Clustering")

        df_clustered = perform_clustering(df_processed.copy())
        plot_clusters(df_clustered)

        # ---------------- FEATURE SELECTION ----------------
        st.header("🎯 Stability Feature Selection")

        if "loan_status" not in df_processed.columns:
            st.error("Column 'loan_status' not found for classification.")
        else:
            stability = stability_feature_selection(df_processed, "loan_status")
            st.dataframe(stability)

            top_features = stability.head(5).index.tolist()

            # ---------------- CLASSIFICATION ----------------
            st.header("🤖 Classification")

            class_results = classification_model(df_processed, "loan_status", top_features)

            for model, res in class_results.items():
                st.subheader(model)
                st.write(res)

        # ---------------- REGRESSION ----------------
        st.header("📈 Regression")

        if "loan_int_rate" not in df_processed.columns:
            st.error("Column 'loan_int_rate' not found for regression.")
        else:
            if not np.issubdtype(df_processed["loan_int_rate"].dtype, np.number):
                st.error("Regression target must be numeric.")
            else:
                reg_results = regression_model(df_processed, "loan_int_rate", top_features)

                for model, res in reg_results.items():
                    st.subheader(model)
                    st.write(res)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
