import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from utils import compute_indicators
from utils_text import build_daily_text_features

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.title("📈 AI Stock Prediction App")
st.write("Educational demo — not financial advice.")

ticker = st.text_input("Enter a stock ticker:", "TSLA")

if st.button("Predict"):
    # Download and preprocess
    px = yf.download(ticker, period="1y", interval="1d")
    px = compute_indicators(px)

    # Load trained model
    model = joblib.load("artifacts/model.joblib")

    # Predict
    preds = model.predict(px.dropna())
    st.line_chart(pd.DataFrame({"Price": px["Close"], "Predicted": preds}, index=px.index))

st.sidebar.info("Upload your own CSV or use built-in tickers.")
