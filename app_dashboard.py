# app_pro_dashboard.py
# Full Professional Dashboard for Next-Day Stock Price Prediction
# Save and run: `streamlit run app_pro_dashboard.py`

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Price Prediction — Pro Dashboard", layout="wide")
# ... your imports
import matplotlib.pyplot as plt 
# Add this line to set the plotting style
plt.style.use('dark_background') 
# ...

import warnings
warnings.filterwarnings("ignore")
# ...
st.title("📊 Stock Prediction Dashboard — Next-Day Close")
# Add hero image below title
st.image(
    "https://wallpapers.com/images/hd/stock-market-monitor-in-sideview-slbivda3m82w17f9.jpg",
    use_container_width=True, # <--- New, recommended parameter
    caption="Stock Market Illustration"
)
st.markdown("---")


# --- Session State Initialization (Global) ---
# Initialize session state for persistent storage across reruns
if 'models_results' not in st.session_state:
    st.session_state['models_results'] = None
if 'results_df' not in st.session_state:
    st.session_state['results_df'] = None
# ---------------------------------------------


# ---------------------------
# Helpers (cached)
# ---------------------------
@st.cache_data
def load_csv_from_path(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    return df.sort_values('Date').reset_index(drop=True)

@st.cache_data
def preprocess_df(df_raw):
    df = df_raw.copy()

    # ensure required columns exist

    required = ['Date','Open','High','Low','Close','Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # make sure numeric
    df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].apply(pd.to_numeric, errors='coerce')

    # Feature engineering per ticker (if exists)

    if 'Ticker' not in df.columns:
        df['Ticker'] = 'SINGLE'
    df = df.sort_values(['Ticker','Date']).reset_index(drop=True)

    def fe(df_group):
        df_group = df_group.copy()
        df_group['Price_Change'] = df_group['Close'] - df_group['Open']
        df_group['High_Low_Range'] = df_group['High'] - df_group['Low']
        df_group['Pct_Change'] = df_group['Close'].pct_change()
        df_group['MA5'] = df_group['Close'].rolling(window=5).mean()
        df_group['MA10'] = df_group['Close'].rolling(window=10).mean()
        df_group['MA20'] = df_group['Close'].rolling(window=20).mean()
        ema12 = df_group['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_group['Close'].ewm(span=26, adjust=False).mean()
        df_group['MACD'] = ema12 - ema26
        delta = df_group['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_group['RSI'] = 100 - (100 / (1 + rs))
        df_group['Close_lag1'] = df_group['Close'].shift(1)
        df_group['Close_lag2'] = df_group['Close'].shift(2)
        df_group['Volume_lag1'] = df_group['Volume'].shift(1)
        df_group['Target'] = df_group['Close'].shift(-1)
        return df_group

    df = df.groupby('Ticker', group_keys=False).apply(fe)
    df = df.dropna().reset_index(drop=True)
    return df

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

@st.cache_data
def select_features(X, threshold=0.95):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Protect essential features from being dropped
    protected = ['Open', 'Close']

    drop_cols = []
    for col in upper.columns:
        if col in protected:
            continue
        if any(upper[col] > threshold):
            drop_cols.append(col)

    X_sel = X.drop(columns=drop_cols)
    return X_sel, drop_cols # CORRECT: Only one return statement is needed

@st.cache_data
def train_models(X_train, y_train, X_test, tune_small=True):
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    
    # Ridge (small grid)
    ridge_params = {'alpha': [0.1,1.0,10.0]} if tune_small else {'alpha':[0.01,0.1,1,10,100]}
    ridge_grid = GridSearchCV(Ridge(), {'alpha': ridge_params['alpha']}, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error', n_jobs=-1)
    ridge_grid.fit(X_train_scaled, y_train)
    ridge = ridge_grid.best_estimator_
    results['Ridge'] = {'model': ridge, 'scaler': scaler, 'y_pred': ridge.predict(X_test_scaled), 'best_params': ridge_grid.best_params_}

    # SVR tuned

    svr_params = {'C':[10,100] if tune_small else [1,10,100,500], 'gamma':['scale','auto',0.01] if tune_small else ['scale','auto',0.001,0.01,0.1], 'epsilon':[0.01,0.1]} 
    svr_grid = GridSearchCV(SVR(kernel='rbf'), svr_params, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error', n_jobs=-1)
    svr_grid.fit(X_train_scaled, y_train.values.ravel())
    svr = svr_grid.best_estimator_
    results['SVR'] = {'model': svr, 'scaler': scaler, 'y_pred': svr.predict(X_test_scaled), 'best_params': svr_grid.best_params_}

    # Gradient Boosting (fast)

    gbr_params = {'n_estimators': [100] if tune_small else [100,300], 'learning_rate':[0.05], 'max_depth':[3,4]}
    gbr_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gbr_params, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error', n_jobs=-1)
    gbr_grid.fit(X_train, y_train)
    gbr = gbr_grid.best_estimator_
    results['Gradient Boosting'] = {'model': gbr, 'scaler': None, 'y_pred': gbr.predict(X_test), 'best_params': gbr_grid.best_params_}

    # Random Forest (fast)

    rf_params = {'n_estimators': [100] if tune_small else [100,200], 'max_depth':[6,8]}
    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), rf_params, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf = rf_grid.best_estimator_
    results['Random Forest'] = {'model': rf, 'scaler': None, 'y_pred': rf.predict(X_test), 'best_params': rf_grid.best_params_}

    # KNN (quick)

    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(X_train_scaled, y_train)
    results['KNN'] = {'model': knn, 'scaler': scaler, 'y_pred': knn.predict(X_test_scaled), 'best_params': None}

    # Decision Tree (quick)
    dtree = DecisionTreeRegressor(random_state=42)
    dtree.fit(X_train, y_train)
    results['Decision Tree'] = {'model': dtree, 'scaler': None, 'y_pred': dtree.predict(X_test), 'best_params': None}

    return results # CORRECT: Return statement must be inside the function

# ---------------------------
# Sidebar - controls
# ---------------------------
st.sidebar.header("Configuration & Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (must include Date,Open,High,Low,Close,Volume). Add 'Ticker' for multi-stock", type=['csv'])
use_example = st.sidebar.checkbox("Use example AMD.csv (default path)", value=True)
default_csv_path = r"C:\Users\ojasv\OneDrive\Documents\AMD.csv"
train_pct = st.sidebar.slider("Train set percentage", 50, 90, 80)
tune_small = st.sidebar.checkbox("Use faster (smaller) hyperparameter search (recommended)", value=True)
feature_corr_thresh = st.sidebar.slider("Correlation threshold to drop features", 0.8, 0.99, 0.95, step=0.01)
run_training = st.sidebar.button("Run training")

# ---------------------------
# Load data
# ---------------------------
try:
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file, parse_dates=['Date'])
    elif use_example:
        df_raw = load_csv_from_path(default_csv_path)
    else:
        st.sidebar.warning("Please upload CSV or check 'Use example'.")
        st.stop()
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

st.sidebar.success(f"Loaded data with shape: {df_raw.shape}")

# ---------------------------
# Main layout - top-row summary
# ---------------------------
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Dataset preview")
    st.dataframe(df_raw.head(10))
with col2:
    st.subheader("Quick stats")
    st.write(f"Rows: {df_raw.shape[0]}  Columns: {df_raw.shape[1]}")
    if 'Ticker' in df_raw.columns:
        st.write(f"Tickers: {df_raw['Ticker'].nunique()}")

# ---------------------------
# Preprocess & feature engineering
# ---------------------------
with st.spinner("Computing features..."):
    try:
        df = preprocess_df(df_raw)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

st.markdown("### Feature summary")
st.write(df[ ['Date','Ticker','Open','High','Low','Close','Volume'] ].head(6))
st.write("Derived features example:")
st.write(df[['Price_Change','High_Low_Range','MA5','MA10','MA20','MACD','RSI']].head(6))

# outlier quick check

with st.expander("Outlier check (IQR)"):
    cols_check = ['Open','High','Low','Close','Volume']
    total_outlier_indices = set()
    for c in cols_check:
        Q1 = df[c].quantile(0.25)
        Q3 = df[c].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        out = df[(df[c] < lower) | (df[c] > upper)]
        st.write(f"{c}: outliers {len(out)} -> {len(out)/len(df):.2%}")
        total_outlier_indices.update(out.index.tolist())
    st.write("Unique outlier rows:", len(total_outlier_indices), "(", len(total_outlier_indices)/len(df), ")")

# ---------------------------
# Prepare X, y, feature selection
# ---------------------------
feature_cols = [
    'Open','High','Low','Close','Volume',
    'Price_Change','High_Low_Range','Pct_Change',
    'MA5','MA10','MA20','MACD','RSI',
    'Close_lag1','Close_lag2','Volume_lag1'
]
X = df[feature_cols].copy()
y = df['Target'].copy()

X_sel, dropped = select_features(X, threshold=feature_corr_thresh)
st.write(f"Dropped features due to high correlation (>{feature_corr_thresh}): {dropped}")
st.write("Using features:", X_sel.columns.tolist())

# ---------------------------
# Train/test split (time series)
# ---------------------------
n_train = int(len(X_sel) * train_pct / 100)
X_train = X_sel.iloc[:n_train].copy()
X_test  = X_sel.iloc[n_train:].copy()
y_train = y.iloc[:n_train].copy()
y_test  = y.iloc[n_train:].copy()

# --- Initialize input Session State variables here (After X_test is defined) ---
if 'user_open_price' not in st.session_state:
    # Use the last open price from the test set as the default
    initial_open = float(X_test['Open'].iloc[-1]) if not X_test.empty else 0.0
    st.session_state['user_open_price'] = initial_open
if 'user_close_price' not in st.session_state:
    # Use the last close price from the test set as the default
    initial_close = float(X_test['Close'].iloc[-1]) if not X_test.empty else 0.0
    st.session_state['user_close_price'] = initial_close
# -----------------------------------------------------------------------------

st.markdown(f"Train / Test sizes: {len(X_train)} / {len(X_test)}")

# ---------------------------
# Training (on demand)
# ---------------------------

if run_training:
    with st.spinner("Training models (this may take a while depending on grid size)..."):
        try:
            # Store trained models directly into session state
            st.session_state['models_results'] = train_models(X_train, y_train, X_test, tune_small=tune_small)
            
            # Build and store the results dataframe immediately
            rows = []
            for name, out in st.session_state['models_results'].items():
                y_pred = out['y_pred']
                mae, mse, rmse, r2 = compute_metrics(y_test, y_pred)
                rows.append({'Model': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'BestParams': out.get('best_params')})
            
            st.session_state['results_df'] = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.session_state['models_results'] = None
            st.session_state['results_df'] = None

# If not run yet, offer quick train button in main area
if st.session_state['models_results'] is None:
    st.info("Press 'Run training' in the sidebar to train models (cached). You can also toggle faster hyperparam search.")
    st.stop()

# ---------------------------
# Build results table (Display only, results_df is set in the training block)
# ---------------------------
st.header("Model Comparison")
st.dataframe(st.session_state['results_df'][['Model','MAE','RMSE','R2','BestParams']])

best_name = st.session_state['results_df'].iloc[0]['Model']
st.success(f"Best model by RMSE: {best_name}")
# Extract the prediction array for the best model
best_pred_arr = st.session_state['models_results'][best_name]['y_pred']

# ---------------------------
# Plots
# ---------------------------
# ---------------------------
# Beautiful Visualization Section
# ---------------------------
st.markdown("## 📈 Model Performance Visualizations")
st.markdown("### A visually enhanced view of model accuracy & predictions")

# Card-style container
st.markdown("""
    <div style="padding:20px; border-radius:15px; background-color:#1e1e1e; border:1px solid #444;">
""", unsafe_allow_html=True)

# R² & RMSE comparison
st.markdown("### 🔷 Model Comparison: R² & RMSE")

fig, ax = plt.subplots(figsize=(11,5))
x = np.arange(len(st.session_state['results_df']))
width = 0.35

# Bar chart
bars1 = ax.bar(x - width/2, st.session_state['results_df']['R2'], width=width, label='R²')
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, st.session_state['results_df']['RMSE'], width=width, alpha=0.6, label='RMSE')

ax.set_xticks(x)
ax.set_xticklabels(st.session_state['results_df']['Model'], rotation=20)
ax.set_ylabel('R²', color='cyan')
ax2.set_ylabel('RMSE', color='orange')
ax.set_title('📊 R² & RMSE Comparison (Enhanced Visualization)', fontsize=15)

# Color highlight
for bar in bars1:
    bar.set_color("cyan")
for bar in bars2:
    bar.set_color("orange")

st.pyplot(fig)
st.markdown("---")

# Actual vs Predicted for Best Model
st.markdown(f"### ⭐ Best Model Prediction: **{best_name}**")

fig2, ax2 = plt.subplots(figsize=(14,4))
ax2.plot(y_test.values, label='Actual', linewidth=2)
ax2.plot(best_pred_arr, label=f'Predicted ({best_name})', linestyle='--', marker='o')
ax2.set_xlabel('Test Index')
ax2.set_ylabel('Close Price')
ax2.legend()
ax2.set_title("📌 Actual vs Predicted — Best Model", fontsize=14)

st.pyplot(fig2)

# Expanders for all models
st.markdown("### 🔍 Compare All Models")
for name, out in st.session_state['models_results'].items():
    with st.expander(f"📌 {name} — Actual vs Predicted"):
        y_pred = out['y_pred']
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(y_test.values, label='Actual', linewidth=2)
        ax.plot(y_pred, label=f'Predicted ({name})', linestyle='--')
        ax.legend()
        ax.set_title(f"{name} — Prediction Performance")
        st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)



# ---------------------------
# User input prediction
# ---------------------------
st.header("Predict Next-Day Close — Use today's Open & Close")
col_a, col_b = st.columns(2)

with col_a:
    user_open = st.number_input(
        "Today's Open", 
        value=st.session_state['user_open_price'], 
        step=0.01, 
        format="%.4f",
        key='user_open_price' # Ensures persistence
    )
with col_b:
    user_close = st.number_input(
        "Today's Close", 
        value=st.session_state['user_close_price'], 
        step=0.01, 
        format="%.4f",
        key='user_close_price' # Ensures persistence
    )

model_choice = st.selectbox("Choose model for prediction (default best):", 
                            st.session_state['results_df']['Model'].tolist(),
                            index=0)

def build_user_features(open_price, close_price, df_hist=df_raw):
    # Build same features using historical df_hist (last ticker only or overall)
    hist = df_hist.sort_values('Date').reset_index(drop=True).copy()
    # If multiple tickers present, pick last ticker present in data
    if 'Ticker' in hist.columns:
        last_ticker = hist['Ticker'].iloc[-1]
        hist = hist[hist['Ticker'] == last_ticker].copy()
    new_row = {
        'Date': pd.to_datetime('today'),
        'Open': open_price,
        'High': max(open_price, close_price),
        'Low': min(open_price, close_price),
        'Close': close_price,
        'Volume': hist['Volume'].iloc[-1]
    }
    df2 = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    # compute features same as preprocess_df.fe
    df2['Price_Change'] = df2['Close'] - df2['Open']
    df2['High_Low_Range'] = df2['High'] - df2['Low']
    df2['Pct_Change'] = df2['Close'].pct_change()
    df2['MA5']  = df2['Close'].rolling(window=5).mean()
    df2['MA10'] = df2['Close'].rolling(window=10).mean()
    df2['MA20'] = df2['Close'].rolling(window=20).mean()
    ema12 = df2['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df2['Close'].ewm(span=26, adjust=False).mean()
    df2['MACD'] = ema12 - ema26
    delta = df2['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df2['RSI'] = 100 - (100 / (1 + rs))
    df2['Close_lag1'] = df2['Close'].shift(1)
    df2['Close_lag2'] = df2['Close'].shift(2)
    df2['Volume_lag1'] = df2['Volume'].shift(1)
    last = df2.iloc[-1]
    feature_values = {c: last[c] for c in X_sel.columns}
    return np.array([feature_values[c] for c in X_sel.columns], dtype=float)

if st.button("Predict next-day close", key="predict_next_day_button"):
    try:
        user_X = build_user_features(user_open, user_close)

        # get selected model and scaler
        model_dict = st.session_state['models_results'][model_choice]
        model_obj = model_dict['model']
        scaler = model_dict.get('scaler', None)

        # scale if needed
        if scaler is not None:
            user_scaled = scaler.transform(user_X.reshape(1, -1))
            predicted_close = model_obj.predict(user_scaled)[0]
        else:
            predicted_close = model_obj.predict(user_X.reshape(1, -1))[0]

        # Tomorrow OPEN = today's CLOSE (user input)
        tomorrow_open = user_close   

        # Display results
        st.success(f"Predicted Next-Day CLOSE Price: {predicted_close:.4f}")
        st.info(f"Tomorrow's OPEN Price (Based on Today's Close): {tomorrow_open:.4f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")




st.write("---")
st.caption(" Hope You Liked :) & Trade Carefully ")
