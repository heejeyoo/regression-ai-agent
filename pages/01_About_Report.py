import os, joblib, numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="About / Report", layout="wide")
st.title("About / Report")

st.markdown("""
**Project:** Multi-Modal Stock Regression (Prices + News Titles)  
**Purpose:** Predict next-day adjusted return using price-based indicators and optional text-mined news features.  
**Disclaimer:** Educational use only.
""")
ART="artifacts"
try:
    cv=pd.read_csv(f"{ART}/cv_results.csv",index_col=0)
    st.subheader("Cross-Validation (mean)"); st.dataframe(cv.style.format({"MAE":"{:.3%}","RMSE":"{:.3%}","R2":"{:.4f}"}))
except Exception:
    st.info("cv_results.csv not found.")
try:
    ts=pd.read_csv(f"{ART}/test_metrics.csv",index_col=0,header=None).squeeze("columns")
    st.subheader("Hold-out Test"); st.dataframe(ts.to_frame("Value").style.format({"Value":"{:.3%}"}))
except Exception:
    st.info("test_metrics.csv not found.")
st.caption("Author: Heeje Yoo • CIS 9660 • Summer 2025")
