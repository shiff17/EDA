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
    color: white;
}
h1, h2, h3, h4, p {
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
st.caption("Ask your data anything 💬")

# -------------------- FILE UPLOAD --------------------
uploaded = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # -------------------- MODE TOGGLE --------------------
    mode = st.radio("Choose Mode", ["🤖 AI Mode", "🎯 Manual Mode"])

    # -------------------- USER QUERY --------------------
    query = st.text_input("💬 Ask a question about your data")

    # -------------------- AI LOGIC --------------------
    def ai_suggest(df, query):
        query = query.lower()

        # Default
        x = None
        y = None
        chart = "bar"

        # GENRE COUNT
        if "genre" in query or "category" in query:
            for col in df.columns:
                if "genre" in col.lower() or df[col].dtype == "object":
                    x = col
                    chart = "bar"
                    break

        # POPULARITY / AVG
        if "average" in query or "highest" in query or "popularity" in query:
            num_cols = df.select_dtypes(include='number').columns
            cat_cols = df.select_dtypes(include='object').columns

            if len(cat_cols) > 0 and len(num_cols) > 0:
                x = cat_cols[0]
                y = num_cols[0]
                chart = "bar"

        return x, y, chart

    # -------------------- MANUAL SELECTION --------------------
    if mode == "🎯 Manual Mode":
        st.subheader("🎯 Select Columns")

        x = st.selectbox("Select Category Column (X-axis)", df.columns)
        y = st.selectbox("Select Numeric Column (Optional)", ["None"] + list(df.columns))
        chart = st.selectbox("Select Chart Type", ["bar", "pie"])

    else:
        x, y, chart = ai_suggest(df, query)
        st.subheader("🤖 AI Suggestions")
        st.write("X-axis:", x)
        st.write("Y-axis:", y if y else "Count")
        st.write("Chart:", chart)

    # -------------------- RUN ANALYSIS --------------------
    if st.button("✨ Analyze"):

        if not x:
            st.warning("AI couldn't determine columns. Try manual mode.")
        else:
            df[x] = df[x].astype(str)

            if y and y != "None" and y in df.columns:
                result = df.groupby(x)[y].mean().reset_index()
                result = result.sort_values(by=y, ascending=False)
            else:
                result = df[x].value_counts().reset_index()
                result.columns = [x, "Count"]
                y = "Count"

            # -------------------- PLOT --------------------
            st.subheader("📊 Visualization")

            if chart == "pie":
                fig = px.pie(result, names=x, values=y)
            else:
                fig = px.bar(result, x=x, y=y, color=x)

            st.plotly_chart(fig, use_container_width=True)

            # -------------------- INSIGHTS --------------------
            st.subheader("🧠 What this means")

            top = result.iloc[0]

            st.success(
                f"👉 The top category is **{top[x]}**, with value **{round(top[y],2)}**.\n\n"
                f"This means this category dominates compared to others in your dataset."
            )

            st.info(
                f"📌 Total categories analyzed: {len(result)}\n"
                f"📊 Highest value: {round(result[y].max(),2)}\n"
                f"📉 Lowest value: {round(result[y].min(),2)}"
            )
