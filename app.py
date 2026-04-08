import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Shiffie's Analytics <3", layout="wide")

# -------------------- DARK CHERRY UI --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #3b0a1a, #7a1f3d);
}
h1, h2, h3, h4, p, label {
    color: white;
}
.stTextInput input {
    background-color: #5a1a2c;
    color: white;
}
.stSelectbox div {
    background-color: #5a1a2c;
}
.stButton>button {
    background-color: #ff4d6d;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("🍒 Shiffie's Analytics <3")
st.caption("Explore your data beautifully ✨")

# -------------------- FILE UPLOAD --------------------
uploaded = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # -------------------- USER QUESTION --------------------
    query = st.text_input("💬 (Optional) What are you trying to find?")

    # -------------------- COLUMN SELECTION --------------------
    st.subheader("🎯 Select Your Analysis")

    x = st.selectbox("Select Category Column (X-axis)", df.columns)
    y = st.selectbox("Select Numeric Column (Optional)", ["None"] + list(df.columns))

    chart = st.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart"])

    # -------------------- ANALYZE --------------------
    if st.button("✨ Analyze"):

        df[x] = df[x].astype(str)

        # -------- AGGREGATION --------
        if y != "None" and y in df.columns:
            result = df.groupby(x)[y].mean().reset_index()
            result = result.sort_values(by=y, ascending=False)
            value_col = y
        else:
            result = df[x].value_counts().reset_index()
            result.columns = [x, "Count"]
            value_col = "Count"

        # -------------------- VISUALIZATION --------------------
        st.subheader("📊 Visualization")

        if chart == "Pie Chart":
            fig = px.pie(result, names=x, values=value_col)
        else:
            fig = px.bar(result, x=x, y=value_col, color=x)

        st.plotly_chart(fig, use_container_width=True)

        # -------------------- INSIGHTS --------------------
        st.subheader("🧠 Insights (Simple Explanation)")

        top = result.iloc[0]

        st.success(
            f"👉 The most important category here is **{top[x]}**, "
            f"with a value of **{round(top[value_col],2)}**."
        )

        st.info(
            f"""
📌 Summary:
- Total categories analyzed: {len(result)}
- Highest value: {round(result[value_col].max(),2)}
- Lowest value: {round(result[value_col].min(),2)}

💡 Interpretation:
This chart shows how different categories compare with each other.
The higher the value, the more dominant that category is.
"""
        )
