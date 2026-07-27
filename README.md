# Stock-Price-Prediction-ML
An interactive Streamlit dashboard for next-day stock closing price prediction using multiple machine learning regression models, technical indicators, feature engineering, and model performance comparison.

📈 Stock Price Prediction using Machine Learning

An interactive Stock Price Prediction Dashboard built with Python, Machine Learning, and Streamlit to predict the next-day closing price of stocks.
The project compares multiple machine learning regression algorithms and provides an interactive dashboard for data preprocessing, feature engineering, model training, performance evaluation, visualization, and next-day stock price prediction.
⚠️ Disclaimer: This project is created for educational and experimental purposes only. Stock market predictions are inherently uncertain and should not be considered financial advice.

🚀 Project Overview

Predicting stock prices is a challenging machine learning problem due to the highly dynamic and volatile nature of financial markets.
This project uses historical stock market data and applies various machine learning regression techniques to predict the next day's closing price.
The application provides a complete machine learning workflow:

📂 Upload historical stock data

🔍 Explore and preview the dataset

🧹 Preprocess financial data

⚙️ Perform feature engineering

📊 Generate technical indicators

🚨 Perform outlier analysis

🔗 Remove highly correlated features

🤖 Train multiple machine learning models

🎯 Perform hyperparameter tuning

📈 Compare model performance

📊 Visualize actual vs predicted prices

🔮 Predict the next-day closing price


Key Features
📂 Data Upload
Users can upload their own stock market dataset in CSV format.

The dataset should contain:
Date

Open

High

Low

Close

Volume

A Ticker column can also be included when working with multiple stocks.

⚙️ Feature Engineering
The project creates several features from historical stock data, including:

-Price Change

-High-Low Range

-Percentage Change

-5-Day Moving Average (MA5)

-10-Day Moving Average (MA10)

-20-Day Moving Average (MA20)

-MACD

-RSI

-Previous Day Closing Price

-Two-Day Lagged Closing Price

-Previous Day Volume

The target variable is the next-day closing price.

🚨 Outlier Detection
The dashboard performs an outlier check using the Interquartile Range (IQR) method.

Outliers are analyzed for important financial features such as:

-Open

-High

-Low

-Close

-Volume

🔗 Feature Selection
Highly correlated features can negatively affect model performance.

The project uses a correlation-based feature selection technique to identify and remove features above a configurable correlation threshold while protecting important features such as Open and Close.

🤖 Machine Learning Models
The project implements and compares multiple regression algorithms:

Model	Description:
Ridge Regression	Regularized linear regression

Support Vector Regression (SVR)	Non-linear regression using support vectors

K-Nearest Neighbors (KNN)	Prediction based on similar observations

Decision Tree	Tree-based regression model

Random Forest	Ensemble of multiple decision trees

Gradient Boosting	Sequential ensemble learning method

