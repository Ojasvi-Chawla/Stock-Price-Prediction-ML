# Next Day Close Stock Price Prediction 
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

🎯 Hyperparameter Tuning :

The project uses GridSearchCV to search for suitable hyperparameters for selected models.

For time-series data, TimeSeriesSplit is used during cross-validation to maintain the chronological nature of the dataset.

This helps avoid randomly mixing future data with past data during model validation.

📊 Model Evaluation
The trained models are evaluated using:

MAE — Mean Absolute Error
MSE — Mean Squared Error
RMSE — Root Mean Squared Error
R² Score — Coefficient of Determination

The models are compared using a performance table, and the best model is selected based on the lowest RMSE.

📈 Visualizations
The Streamlit dashboard provides several visualizations, including:

R² Score comparison
RMSE comparison
Actual vs Predicted prices
Best model prediction performance
Individual model prediction comparisons

These visualizations make it easier to understand and compare model performance.

🔮 Next-Day Stock Price Prediction
The dashboard allows users to enter:

Today's Opening Price
Today's Closing Price

The selected machine learning model then predicts the next-day closing price.

The application also estimates the next day's opening price based on the entered current closing price.

🛠️ Technologies Used :

Python

Pandas — Data manipulation and analysis

NumPy — Numerical computation

Matplotlib — Data visualization

Scikit-learn — Machine learning

Streamlit — Interactive web dashboard

Machine Learning Libraries

Ridge

SVR

KNeighbors Regressor

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Grid Search CV

Time Series Split

Robust Scaler


⚙️ Installation 

1. Clone the Repository
git clone https://github.com/Ojasvi Chawla/Stock-Price-Prediction-ML.git

3. Navigate to the Project Directory
cd Stock-Price-Prediction-ML

4. Install Required Libraries
pip install -r requirements.txt

Or install the dependencies manually:
pip install streamlit pandas numpy matplotlib scikit-learn


▶️ Run the Application
-Run the Streamlit application using:
-streamlit run app_pro_dashboard.py
-The dashboard will open in your browser.


🖥️ How to Use
Step 1: Upload Dataset
Upload a CSV file containing historical stock market data.

Step 2: Configure Parameters
Use the sidebar to configure:

Training dataset percentage
Hyperparameter tuning settings
Feature correlation threshold
Step 3: Train Models

Click Run Training to train and evaluate the machine learning models.

Step 4: Compare Models
Review the model performance based on:

MAE
RMSE
R² Score

Step 5: Select a Model
Choose a model from the available trained models.

Step 6: Predict Next-Day Close
Enter today's:

Open price
Close price

Click Predict Next-Day Close to generate the prediction.


📊 Machine Learning Workflow
Historical Stock Data
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Technical Indicators
(MA, MACD, RSI, Lag Features)
        │
        ▼
Outlier Analysis
        │
        ▼
Correlation-Based Feature Selection
        │
        ▼
Time-Series Train/Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Model Training
        │
        ├── Ridge Regression
        ├── SVR
        ├── KNN
        ├── Decision Tree
        ├── Random Forest
        └── Gradient Boosting
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Model Comparison
        │
        ▼
Best Model Selection
        │
        ▼
Next-Day Closing Price Prediction


📌 Project Highlights
-Implemented multiple machine learning regression algorithms
-Used time-series-based train/test splitting
-Applied RobustScaler for feature scaling
-Implemented feature engineering using financial indicators
-Used GridSearchCV for hyperparameter optimization
-Compared models using multiple evaluation metrics
-Developed an interactive Streamlit dashboard
-Added user-input-based next-day stock price prediction
