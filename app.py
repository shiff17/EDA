import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Shiffie's Analytics <3", layout="wide")

# -------------------- CLEAN PINK UI --------------------
st.markdown("""
<style>
.stApp {
    background-color: #fff5f7;
}
h1, h2, h3 {
    color: #c2185b;
}
.stButton>button {
    background-color: #ec407a;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("💗 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Categorical Analysis"])

# -------------------- HOME --------------------
if page == "Home":
    st.title("💗 Shiffie's Analytics <3")
    st.write("✨ Upload your dataset and explore categorical insights with AI support")

# -------------------- CATEGORICAL PAGE --------------------
elif page == "Categorical Analysis":

    st.title("📊 Categorical Analysis")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # -------- DETECT CATEGORICAL COLUMNS --------
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        # -------- AI SUGGESTION --------
        st.subheader("🤖 AI Suggestions")

        if len(cat_cols) > 0:
            suggested_col = cat_cols[0]
            st.write(f"💡 Suggested column for analysis: **{suggested_col}**")
        else:
            st.warning("No categorical columns detected.")
            suggested_col = None

        # -------- MANUAL SELECTION --------
        st.subheader("🎯 Select Column")

        selected_col = st.selectbox(
            "Choose a categorical column",
            df.columns
        )

        # -------- ANALYZE BUTTON --------
        if st.button("Analyze"):

            # Convert to string to avoid errors
            df[selected_col] = df[selected_col].astype(str)

            counts = df[selected_col].value_counts().reset_index()
            counts.columns = [selected_col, "Count"]

            # -------- VISUALIZATION --------
            st.subheader(f"📊 Distribution of {selected_col}")

            fig = px.bar(
                counts,
                x=selected_col,
                y="Count",
                color=selected_col,
                title=f"{selected_col} Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

            # -------- PIE CHART --------
            st.subheader("🥧 Pie Chart")

            fig2 = px.pie(
                counts,
                values="Count",
                names=selected_col
            )

            st.plotly_chart(fig2, use_container_width=True)

            # -------- INSIGHTS --------
            st.subheader("🧠 Quick Insights")

            top = counts.iloc[0]
            st.success(
                f"Most frequent category: **{top[selected_col]}** "
                f"with {top['Count']} occurrences."
            )
            
