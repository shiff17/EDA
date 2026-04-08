# 🔥 FINAL OVERVIEW BOX (FIXED + DYNAMIC)

# Safe defaults
top_group = "Run group analysis"
ml_score_text = "Run ML section"
best_model_name = "Run ML section"

# Try to compute real top group (if group analysis possible)
if cat_cols and numeric_cols:
    try:
        sample_group = cat_cols[0]
        sample_value = numeric_cols[0]
        grp = df.groupby(sample_group)[sample_value].mean().sort_values(ascending=False)
        top_group = grp.index[0]
    except:
        pass

# Try to capture ML results if they exist
if 'result_df' in locals():
    best_row = result_df.loc[result_df['Score'].idxmax()]
    best_model_name = best_row['Model']
    ml_score_text = f"{best_row['Score']:.3f}"

# Dataset readiness
dataset_status = "ANALYSIS READY" if missing_pct < 5 else "PARTIALLY READY"
clean_status = "✅ READY" if missing_pct < 5 else "⚠️ CLEAN NEEDED"

st.markdown(f"""
<div class="overview-box">
<h2 style='color: #ff1493; margin-bottom: 1rem;'>🎯 COMPLETE ANALYTICS SUMMARY</h2>

<h3>📊 Dataset Status</h3>
- <b>Rows</b>: {len(df):,} | <b>Columns</b>: {len(df.columns)} | <b>Missing</b>: {missing_pct:.1f}%  
{clean_status}

<h3>🔗 Key Findings</h3>
- <b>Strongest correlation</b>: Check heatmap for predictors  
- <b>Top business group</b>: {top_group}  
- <b>Prediction power</b>: {ml_score_text}

<h3>🚀 Action Plan</h3>
1. <b>Clean</b> {missing_pct:.1f}% missing data  
2. <b>Focus</b> on top correlated features  
3. <b>Deploy</b> best model ({best_model_name})  
4. <b>Monitor</b> key business metrics  

<h3 style='color: #ff1493;'>💡 YOUR DATASET IS {dataset_status}!</h3>
</div>
""", unsafe_allow_html=True)
