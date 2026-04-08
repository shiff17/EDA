import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# -------------------- 💗 PINK THEME --------------------
st.set_page_config(page_title="Shiffie's Analytics <3", layout="wide")

st.markdown("""
<style>
body {
    background-color: #fff0f5;
}
.stApp {
    background: linear-gradient(135deg, #ffe6f0, #fff0f5);
}
h1, h2, h3 {
    color: #d63384;
}
.stButton>button {
    background-color: #ff69b4;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- CLEAN DATA --------------------
def clean_data(df):
    df = df.copy()

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Encode categorical
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))

    return df


# -------------------- AI CHAT LOGIC --------------------
def ai_chat_analysis(df, query):
    st.subheader("🤖 AI Insights")

    query = query.lower()

    # -------- CATEGORICAL --------
    if "categorical" in query:
        cat_cols = df.select_dtypes(include=['object']).columns

        if len(cat_cols) == 0:
            st.warning("No categorical columns found.")
            return

        col = cat_cols[0]
        st.write(f"📊 Analyzing categorical column: **{col}**")

        fig = px.bar(df[col].value_counts())
        st.plotly_chart(fig, use_container_width=True)

    # -------- CLASSIFICATION --------
    elif "classification" in query or "fraud" in query:
        target = df.columns[-1]

        st.write(f"🎯 Target selected: **{target}**")

        data = clean_data(df)

        X = data.drop(target, axis=1)
        y = data[target]

        model = RandomForestClassifier()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text(classification_report(y_test, y_pred))

    # -------- REGRESSION --------
    elif "sales" in query or "regression" in query:
        target = df.select_dtypes(include=np.number).columns[-1]

        st.write(f"📈 Predicting: **{target}**")

        data = clean_data(df)

        X = data.drop(target, axis=1)
        y = data[target]

        model = RandomForestRegressor()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

    # -------- DEFAULT --------
    else:
        st.info("Try: 'categorical analysis', 'fraud detection', 'sales prediction'")


# -------------------- UI --------------------
st.title("💗 Shiffie's Analytics <3")
st.caption("Your AI-powered data bestie ✨")

uploaded = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # 💬 CHAT INPUT
    query = st.text_input("💬 Ask me what to do with your data (e.g. 'categorical analysis', 'predict sales')")

    if query:
        ai_chat_analysis(df, query)
