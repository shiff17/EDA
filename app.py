import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# -------------------- UNIVERSAL RISK SCORING --------------------
def compute_universal_risk(df):
    numeric_cols = df.select_dtypes(include=['number']).columns

    if len(numeric_cols) == 0:
        df["cvss"] = 0
        df["severity"] = "Low"
        df["status"] = "Safe"
        return df

    scaler = MinMaxScaler()
    norm = scaler.fit_transform(df[numeric_cols])

    risk_score = norm.mean(axis=1)
    df["cvss"] = risk_score * 10

    df["severity"] = pd.cut(
        df["cvss"],
        bins=[0, 3.9, 6.9, 8.9, 10],
        labels=["Low", "Medium", "High", "Critical"]
    )

    df["status"] = df["severity"].astype(str).apply(
        lambda x: "Vulnerable" if x in ["High", "Critical"] else "Safe"
    )

    return df


# -------------------- AI MODEL ASSISTANT --------------------
def ai_model_assistant(df):
    st.subheader("🤖 AI Model Assistant")

    keywords = ["target", "label", "fraud", "default", "risk", "status"]
    possible_targets = [col for col in df.columns if any(k in col.lower() for k in keywords)]

    target_col = possible_targets[0] if possible_targets else df.columns[-1]

    task = "Classification" if df[target_col].nunique() <= 10 else "Regression"
    feature_cols = [col for col in df.columns if col != target_col]

    st.write("### 🧠 AI Suggestions")
    st.write("**Task:**", task)
    st.write("**Target Column:**", target_col)
    st.write("**Feature Columns:**", feature_cols)

    if st.button("⚡ Run AI Suggested Model"):

        data = df[feature_cols + [target_col]].copy()

        le = LabelEncoder()
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = le.fit_transform(data[col].astype(str))

        X = data[feature_cols]
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier() if task == "Classification" else RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📊 Model Results")

        if task == "Classification":
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.text(classification_report(y_test, y_pred))
        else:
            st.write("MSE:", mean_squared_error(y_test, y_pred))
            st.write("R2 Score:", r2_score(y_test, y_pred))

        importance = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.subheader("📌 Feature Importance")
        st.dataframe(imp_df, use_container_width=True)


# -------------------- VISUALIZATION FUNCTION --------------------
def ml_visualizations(df):
    st.header("📊 ML-Based Insights")

    if "severity" in df.columns:
        sev_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        df["severity_num"] = df["severity"].astype(str).map(sev_map).fillna(0)

    vis_type = st.selectbox("Choose visualization", ["Bar", "Pie", "Scatter", "Histogram"])

    if vis_type == "Bar" and "severity" in df.columns:
        st.plotly_chart(px.bar(df["severity"].value_counts()), use_container_width=True)

    elif vis_type == "Pie" and "status" in df.columns:
        st.plotly_chart(px.pie(df, names="status"), use_container_width=True)

    elif vis_type == "Scatter" and "severity_num" in df.columns:
        st.plotly_chart(px.scatter(df, y="severity_num"), use_container_width=True)

    elif vis_type == "Histogram" and "severity" in df.columns:
        st.plotly_chart(px.histogram(df, x="severity"), use_container_width=True)


# -------------------- NAVIGATION --------------------
st.sidebar.title("🛡 MESS")
page = st.sidebar.radio("Navigate", ["Homepage", "Analytics", "Visualization"])


# -------------------- HOMEPAGE --------------------
if page == "Homepage":
    st.title("🛡 MESS: Machine-driven Exploit Shielding System")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # 🔥 AI FEATURE HERE
        ai_model_assistant(df)

        df = compute_universal_risk(df)

        st.subheader("📄 Data Preview")
        st.dataframe(df)

        df = df.dropna()

        if "severity" in df.columns:
            df["severity_num"] = df["severity"].map({"Low":1,"Medium":2,"High":3,"Critical":4})

            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            df["cluster"] = km.fit_predict(df[["severity_num"]])

        st.subheader("✨ Processed Data")
        st.dataframe(df)


# -------------------- ANALYTICS --------------------
elif page == "Analytics":
    st.title("📊 Analytics")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # 🔥 AI FEATURE HERE TOO
        ai_model_assistant(df)

        df = compute_universal_risk(df)

        st.write(df.describe())

        if "status" in df.columns:
            st.plotly_chart(px.pie(df, names="status"), use_container_width=True)


# -------------------- VISUALIZATION --------------------
elif page == "Visualization":
    st.title("📈 Visualization")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        df = compute_universal_risk(df)
        ml_visualizations(df)
