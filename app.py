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
st.caption("Robust ML + Clean Insights + No Errors ✨")

# -------------------- PREPROCESSING --------------------
def preprocess_data(df):
    df = df.copy()

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Encode categorical
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors='coerce')
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
    scatter = ax.scatter(comp[:, 0], comp[:, 1], c=df["cluster"], cmap='viridis')
    ax.set_title("Cluster Visualization (PCA)")
    plt.colorbar(scatter)
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

    return (scores / iterations).sort_values(ascending=False)


# -------------------- CLASSIFICATION --------------------
def classification_model(df, target, features):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
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
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test, pred)
        }

    return results


# -------------------- REGRESSION --------------------
def regression_model(df, target, features):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42)
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

    st.success(f"🎯 Highest count category: **{top}** ({counts.max()})")
    st.info(f"📉 Lowest count category: **{bottom}** ({counts.min()})")
    st.caption("This represents frequency (count), not averages.")


# -------------------- MAIN --------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Show data info
    st.subheader("📊 Data Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    
    st.text(f"Columns: {', '.join(df.columns.tolist())}")

    try:
        df_processed = preprocess_data(df)

        # ---------------- CATEGORICAL ----------------
        st.header("📊 Categorical Analysis")

        cat_cols = df.select_dtypes(include='object').columns.tolist()

        if len(cat_cols) == 0:
            st.warning("⚠️ No categorical columns found.")
        else:
            col = st.selectbox("Select categorical column", cat_cols)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=df[col].astype(str), ax=ax, palette='viridis')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            generate_insights(df, col)

        # ---------------- FIXED AGGREGATED ANALYSIS ----------------
        st.header("📊 Aggregated Analysis")

        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            x = st.selectbox("Select X (Category) Column", all_cols)
        with col2:
            y_options = ["None"] + numeric_cols
            y = st.selectbox("Select Y (Numeric) Column", y_options)

        if st.button("Run Analysis", type="primary"):
            df_analysis = df.copy()
            df_analysis[x] = df_analysis[x].astype(str)
            
            if y != "None":
                # Verify Y is actually numeric
                if y not in numeric_cols:
                    st.error(f"❌ '{y}' is not a numeric column!")
                    st.stop()
                
                # Safe numeric conversion and mean calculation
                df_analysis[y] = pd.to_numeric(df_analysis[y], errors='coerce')
                df_analysis = df_analysis.dropna(subset=[y])
                
                if len(df_analysis) == 0:
                    st.error("❌ No valid numeric data found for analysis!")
                    st.stop()
                
                result = df_analysis.groupby(x)[y].agg(['mean', 'count']).reset_index()
                result.columns = [x, 'mean_value', 'count']
                result = result.sort_values('mean_value', ascending=False)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=result.head(10), x=x, y='mean_value', ax=ax, palette='viridis')
                plt.xticks(rotation=45)
                plt.title(f'Average {y} by {x}')
                plt.tight_layout()
                st.pyplot(fig)

                st.success(f"🎯 Highest average **{y}**: **{result.iloc[0][x]}** ({round(result.iloc[0]['mean_value'], 2)})")
                st.dataframe(result.head(10))

            else:
                # Count analysis only
                counts = df_analysis[x].value_counts().reset_index()
                counts.columns = [x, "Count"]
                counts = counts.sort_values("Count", ascending=False)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=counts.head(10), x=x, y="Count", ax=ax, palette='viridis')
                plt.xticks(rotation=45)
                plt.title(f'Count of {x}')
                plt.tight_layout()
                st.pyplot(fig)

                st.success(f"🎯 Most frequent category: **{counts.iloc[0][x]}** ({counts.iloc[0]['Count']})")
                st.dataframe(counts.head(10))

        # ---------------- CLUSTERING ----------------
        st.header("🧠 Clustering Analysis")
        
        if st.button("Run Clustering", type="secondary"):
            df_clustered = perform_clustering(df_processed.copy())
            plot_clusters(df_clustered)
            
            st.subheader("Cluster Distribution")
            st.dataframe(df_clustered['cluster'].value_counts().reset_index())

        # ---------------- FEATURE SELECTION & ML ----------------
        st.header("🎯 ML Analysis (Auto-detect targets)")
        
        # Auto-detect common ML targets
        possible_targets = []
        if "loan_status" in df_processed.columns:
            possible_targets.append("loan_status")
        if "loan_int_rate" in df_processed.columns:
            possible_targets.append("loan_int_rate")
        if "target" in df_processed.columns:
            possible_targets.append("target")
            
        if possible_targets:
            target = st.selectbox("Select target column", possible_targets)
            
            if st.button("Run ML Pipeline", type="primary"):
                stability = stability_feature_selection(df_processed, target)
                st.subheader("🔥 Top Stable Features")
                st.dataframe(stability.head(10))
                
                top_features = stability.head(min(10, len(stability))).index.tolist()
                
                if len(top_features) > 0:
                    # Classification if target seems binary-ish
                    unique_vals = len(df_processed[target].unique())
                    if unique_vals <= 10:  # Likely classification
                        st.header("🤖 Classification Results")
                        try:
                            class_results = classification_model(df_processed, target, top_features)
                            for model, res in class_results.items():
                                st.subheader(f"**{model}**")
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Accuracy", f"{res['Accuracy']:.3f}")
                                col2.metric("Precision", f"{res['Precision']:.3f}")
                                col3.metric("Recall", f"{res['Recall']:.3f}")
                                col4.metric("F1", f"{res['F1 Score']:.3f}")
                                st.write("Confusion Matrix:")
                                st.write(res['Confusion Matrix'])
                        except Exception as e:
                            st.error(f"Classification failed: {e}")
                    
                    # Always try regression
                    st.header("📈 Regression Results")
                    try:
                        reg_results = regression_model(df_processed, target, top_features)
                        for model, res in reg_results.items():
                            st.subheader(f"**{model}**")
                            col1, col2 = st.columns(2)
                            col1.metric("RMSE", f"{res['RMSE']:.3f}")
                            col2.metric("R²", f"{res['R2 Score']:.3f}")
                    except Exception as e:
                        st.error(f"Regression failed: {e}")
        else:
            st.info("ℹ️ No common ML target columns found (loan_status, loan_int_rate, target).")

    except Exception as e:
        st.error(f"❌ Something went wrong: {str(e)}")
        st.error("Please check your data format and try again.")
