# NVIDIA Stock Movement Prediction h1
Abstract
This project analyzes NVIDIA's stock prices from 2021 to 2023 and aims to predict future stock trends and prices using engineered features and machine learning techniques.

Introduction
Stock market prediction is crucial for informed financial decisions. This report uses NVIDIA's stock data from Yahoo Finance, focusing on features like Open, High, Low, Close, Volume (OHLCV), and technical indicators such as MACD, RSI, and Bollinger Bands. We evaluate the effectiveness of machine learning models against standard engineered features in predicting stock trends.

Dataset and Features
Period: 2021-2023
Source: Yahoo Finance
Main Features: OHLCV, Adjusted Close, and Date
Engineered Features: MACD, Signal Line, RSI, Bollinger Bands
Additional Features: Reported Earnings, Reported Estimate, Surprise
Research Questions
Engineered Features: Are technical indicators accurate in showing stock trends?
Machine Learning Models: How reliable are models like RF, SVR, and LSTM in forecasting using historical data?
Comparison: How do ML models fare against engineered features in terms of prediction accuracy?
Assumptions and Hypotheses
The dataset is accurate and representative.
Past stock behavior is indicative of future performance.
Hypotheses on daily returns, Bollinger Bands, and MACD signals are tested for market trend indications.
Methodology
Statistical analysis of NVIDIA stock data.
Evaluation of stock momentum indicators (Bollinger Bands, RSI, MACD).
Machine Learning Models: RF, SVR, XGBoost, LSTM.
Use of time series data for model training and testing.
Results
Statistical Analysis: The stock shows a positive average daily return with notable volatility.
Engineered Features: Indicators like Bollinger Bands and MACD provided useful trend information.
Machine Learning Models: XGBoost showed promising results, while RF and SVR had limitations in handling continuous data and small fluctuations, respectively.
Conclusion
The study provides insights into the predictive power of both engineered features and machine learning models. It highlights the strengths and limitations of various approaches to stock market prediction.

Keywords
#NVIDIA #StockPrediction #MachineLearning #TechnicalAnalysis #TimeSeries
