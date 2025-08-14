# Stock Price Prediction AI Agent with News Sentiment Analysis

## 📈 Project Overview

This project is a regression-based AI agent that predicts stock prices by combining traditional financial data with real-time news sentiment analysis. The agent is deployed as an interactive web application using Streamlit, allowing users to select a stock ticker and receive a price prediction based on a machine learning model.

The core of this project is its unique feature engineering process. It fetches daily stock prices (`Open`, `High`, `Low`, `Close`, `Volume`) from the **Yahoo Finance API** and simultaneously pulls relevant news headlines from the **NewsAPI**. It then calculates a daily sentiment score and news volume from the headlines, creating a richer feature set for the regression models.

This demonstrates an end-to-end machine learning workflow, from data integration and feature engineering to model training, evaluation, and deployment.

## 🚀 Live Application

**You can access the live, deployed application here:**

[https://[YOUR-STREAMLIT-APP-URL].streamlit.app/](https://regression-ai-agent-apnx6mdtea9hcnmrygkzc4.streamlit.app/)

## ✨ Features

* **Interactive Prediction**: Enter any valid stock ticker to get a real-time price prediction.
* **Sentiment-Enhanced Model**: The prediction isn't just based on numbers; it's influenced by the sentiment of recent news, providing a more holistic analysis.
* **Model Insights Dashboard**: Visualize the stock's historical price alongside the calculated daily news sentiment to understand the factors influencing the prediction.
* **Dual Model Comparison**: The backend trains and evaluates both a `Linear Regression` model and a more powerful `XGBoost` model, with the champion model used for final predictions.
* **Professional UI**: A clean, modern interface built with Streamlit, featuring a dark theme suitable for financial applications.

## 🛠️ Technical Stack

* **Programming Language**: Python 3.11
* **Data Acquisition**:
    * `yfinance`: For historical stock market data.
    * `newsapi-python`: For fetching news articles.
* **Data Processing & Modeling**:
    * `pandas`: For data manipulation and analysis.
    * `scikit-learn`: For data splitting and the Linear Regression model.
    * `xgboost`: For the Gradient Boosting model.
    * `nltk (VADER)`: For calculating news headline sentiment.
* **Web Framework & Deployment**:
    * `streamlit`: For building the interactive web application.
    * `Streamlit Community Cloud`: For 24/7 hosting.
* **Visualization**:
    * `plotly`: For creating interactive charts.

## ⚙️ Setup and Local Installation

To run this project on your local machine, please follow these steps:

**1. Clone the Repository**
```bash
git clone [https://github.com/](https://github.com/)[heejeyoo]/[regression-ai-agent].git
cd [regression-ai-agent]
