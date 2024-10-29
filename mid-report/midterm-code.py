import yfinance as yf
import pandas as pd
from datetime import datetime

""" Import Dataset and Define Dataframe"""
# add VIX, SP500, IXIC, DJI, HSI, DX=F, GC=F, CL=F from yfinance
start = datetime(2003, 1, 1)
end = datetime(2024, 1, 1)

tickers = "^VIX ^GSPC ^IXIC ^DJI DX=F ^HSI GC=F CL=F"
data = yf.download(tickers, start=start, end=end)

selected_data = pd.DataFrame(index=data.index)


selected_data['VIX Open'] = data['Open']['^VIX']
selected_data['VIX High'] = data['High']['^VIX']
selected_data['VIX Low'] = data['Low']['^VIX']
selected_data['VIX Close'] = data['Close']['^VIX']


for ticker in ['^GSPC', '^IXIC', '^DJI', '^HSI', 'DX=F', 'GC=F', 'CL=F']:
    if ticker in data['Close'].columns:
        selected_data[f'{ticker} Close'] = data['Close'][ticker]
        selected_data[f'{ticker} Volume'] = data['Volume'][ticker]


# add macroeconomic indicators from pandas_datareader
import pandas_datareader.data as web

effr = web.DataReader('EFFR', 'fred', start, end)

macro_data = pd.concat([effr], axis=1)
macro_data.columns = ['EFFR']

# Unified time format
selected_data.index = pd.to_datetime(selected_data.index).tz_localize(None)

full_data = pd.concat([selected_data, macro_data], axis=1, join='inner')

# calculate and add technology indicators
import pandas_ta as ta

vix_data = yf.download('^VIX', start=start, end=end)

# MACD
macd = ta.macd(close=vix_data['Close'], fast=12, slow=26, signal=9)
# ATR
atr = ta.atr(high=vix_data['High'], low=vix_data['Low'], close=vix_data['Close'], length=14)
# RSI
rsi = ta.rsi(close=vix_data['Close'], length=14)

full_data = pd.concat([full_data, macd[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']], pd.Series(atr, name='ATR'), pd.Series(rsi, name='RSI')], axis=1)
full_data = full_data.loc[:,~full_data.columns.duplicated()] # Remove Duplicate Columns

""" Data Cleaning"""
# If the value is NaN, then fill with the forward value, or backward if still NaN
cleaned_data = full_data.fillna(method='ffill').fillna(method='bfill')

"""Descriptive analysis"""
cleaned_data.info()
cleaned_data.describe()

# plot the trend of VIX Price Data
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(14, 7))
plt.plot(cleaned_data['VIX Close'], label='VIX Close', color='black')
plt.title('VIX Price Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Histogram
# visualize the distribution of data, identify its shape, detect outliers
import seaborn as sns
import matplotlib.pyplot as plt

num_cols = cleaned_data.select_dtypes(include=['number']).columns
n_cols = 4
n_rows = (len(num_cols) + n_cols - 1) // n_cols

plt.figure(figsize=(20, 5 * n_rows))

for i, column in enumerate(num_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(cleaned_data[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Box plot
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(20, 30))

num_columns = len(cleaned_data.columns)

num_rows = num_columns // 3 + 1 if num_columns % 3 != 0 else num_columns // 3

fig, axs = plt.subplots(num_rows, 3, figsize=(18, num_rows * 6))

for i, col in enumerate(cleaned_data.columns):
    ax = axs[i // 3, i % 3]
    sns.boxplot(x=cleaned_data[col], ax=ax)
    ax.set_title(f'Boxplot of {col}')
    ax.set_xlabel('')
    ax.set_ylabel('Value')

plt.tight_layout()
plt.show()

#Scatter plot
import seaborn as sns
import matplotlib.pyplot as plt

features = [
    'VIX Close', 'VIX Open', 'VIX High', 'VIX Low',
    '^GSPC Close', '^GSPC Volume', '^IXIC Close', '^IXIC Volume',
    '^DJI Close', '^DJI Volume', '^HSI Close', '^HSI Volume',
    'DX=F Close', 'DX=F Volume', 'GC=F Close', 'GC=F Volume',
    'CL=F Close', 'CL=F Volume', 'EFFR', 'MACD_12_26_9',
    'MACDs_12_26_9', 'MACDh_12_26_9', 'ATR', 'RSI'
]

sns.pairplot(cleaned_data[features])
plt.suptitle('Scatter Plot Matrix of Selected Features', size=16)
plt.show()

# Feature correlations
corr = cleaned_data.corr()

plt.figure(figsize=(18, 15))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Financial Data')
plt.show()

"""Outlier Detection"""
from sklearn.ensemble import IsolationForest

# method1 - IQR method - The reason why chooses the method is that the distributions are not all normal distribution.

# calculate Q1, Q3, IQR
Q1 = cleaned_data.quantile(0.25)
Q3 = cleaned_data.quantile(0.75)
IQR = Q3 - Q1

# choose the boundary for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# detect the outliers
outliers_iqr = (cleaned_data < lower_bound) | (cleaned_data > upper_bound)

# delete outliers
cleaned_data1 = cleaned_data[~(outliers_iqr.any(axis=1))]

# method2 - Isolation Forest
iso = IsolationForest(contamination=0.05)
outliers_iso = iso.fit_predict(cleaned_data1)
cleaned_data1['is_outlier'] = outliers_iso == -1

# split feature and target
target = cleaned_data1['VIX Close']
features = cleaned_data1.drop(columns=['VIX Close', 'is_outlier'])

"""Models"""
# import the necessary libraries
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor
import optuna

class VIXPredictionFramework:
    def __init__(self, data, target_col='VIX Close', test_size=0.2):
        self.data = data
        self.target_col = target_col
        self.test_size = test_size
        self.models = {}
        self.results = {}

    def prepare_time_series_data(self, seq_length=10):
        """Time series data preparation, including rolling window creation"""
        # standardized data
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.data)

        # create a time series feature
        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:(i + seq_length)])
            y.append(scaled_data[i + seq_length, self.data.columns.get_loc(self.target_col)])

        X = np.array(X)
        y = np.array(y)

        # split train and test datasets
        train_size = int(len(X) * (1 - self.test_size))
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_garch(self, p=1, q=1):
        """train GARCH model"""
        try:
            # GARCH modeling using closing prices of raw data - write comments about how the model works
            returns = 100 * self.data[self.target_col].pct_change().dropna()
            model = arch_model(returns, p=p, q=q)
            self.models['GARCH'] = model.fit(disp='off')
            return self.models['GARCH']
        except Exception as e:
            print(f"GARCH model training failed: {str(e)}")
            return None

    def train_arima(self, order=(1,1,1)):
        """train ARIMA model"""
        try:
            model = ARIMA(self.data[self.target_col], order=order)
            self.models['ARIMA'] = model.fit()
            return self.models['ARIMA']
        except Exception as e:
            print(f"ARIMA model training failed: {str(e)}")
            return None

# Using Optuna to optimize the hyperparameters of LSTM and XGBoost models.
# Optuna iteratively optimizes the hyperparameters based on the output of the objective function,
# such as validation loss.

    def train_lstm(self, optimizer_trial=None):
        """train LSTM model, support Optuna optimization"""
        try:
            # Get hyperparameters (can be optimized via Optuna)
            if optimizer_trial:
                units = optimizer_trial.suggest_int('lstm_units', 32, 256)
                layers = optimizer_trial.suggest_int('lstm_layers', 1, 3)
                dropout = optimizer_trial.suggest_float('dropout', 0.1, 0.5)
                learning_rate = optimizer_trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            else:
                units, layers, dropout, learning_rate = 64, 2, 0.2, 0.001

            model = Sequential()
            model.add(LSTM(units, return_sequences=True if layers > 1 else False,
                         input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
            model.add(Dropout(dropout))

            for i in range(layers-2):
                model.add(LSTM(units, return_sequences=True))
                model.add(Dropout(dropout))

            if layers > 1:
                model.add(LSTM(units))
                model.add(Dropout(dropout))

            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

            self.models['LSTM'] = model
            history = model.fit(self.X_train, self.y_train,
                              validation_split=0.2,
                              epochs=50,
                              batch_size=32,
                              verbose=0)
            return history
        except Exception as e:
            print(f"LSTM model training failed: {str(e)}")
            return None

    def train_xgboost(self, optimizer_trial=None):
        """train XGBoost model, support Optuna optimization"""
        try:
            if optimizer_trial:
                params = {
                    'max_depth': optimizer_trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': optimizer_trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                    'n_estimators': optimizer_trial.suggest_int('n_estimators', 100, 1000),
                    'min_child_weight': optimizer_trial.suggest_int('min_child_weight', 1, 7),
                    'subsample': optimizer_trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': optimizer_trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            else:
                params = {
                    'max_depth': 6,
                    'learning_rate': 0.01,
                    'n_estimators': 500
                }

            model = XGBRegressor(**params)
            # Reshape data to suit XGBoost
            X_train_2d = self.X_train.reshape(self.X_train.shape[0], -1)
            X_test_2d = self.X_test.reshape(self.X_test.shape[0], -1)

            self.models['XGBoost'] = model.fit(X_train_2d, self.y_train)
            return model
        except Exception as e:
            print(f"XGBoost model training failed: {str(e)}")
            return None

    def optimize_hyperparameters(self, model_name, n_trials=100):
        """Optimizing model hyperparameters using Optuna"""
        def objective(trial):
            if model_name == 'LSTM':
                history = self.train_lstm(trial)
                return history.history['val_loss'][-1]
            elif model_name == 'XGBoost':
                model = self.train_xgboost(trial)
                X_test_2d = self.X_test.reshape(self.X_test.shape[0], -1)
                predictions = model.predict(X_test_2d)
                return mean_squared_error(self.y_test, predictions)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def _calculate_rmse(self, y_true, y_pred):
        """Calculate Root Mean Square Error (RMSE)"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def _calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error (MAPE)"""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def _calculate_directional_accuracy(self, y_true, y_pred):
        """Calculate Directional Accuracy"""
        return np.mean((np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1]))) * 100

    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0):
        """Calculate Sharpe Ratio based on predicted returns"""
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)

    def evaluate_model(self, model_name):
        """evaluate the performance of the models"""
        try:
            y_test = self.y_test
            
            if model_name == 'GARCH':
                forecast = self.models[model_name].forecast(horizon=len(self.y_test))
                predictions = forecast.mean.values[-1]
                if len(predictions) != len(self.y_test):
                    predictions = predictions[:len(self.y_test)]
            elif model_name == 'ARIMA':
                predictions = self.models[model_name].forecast(len(self.y_test))
                if len(predictions) != len(self.y_test):
                    if len(predictions) < len(self.y_test):
                        y_test = self.y_test[:len(predictions)]
                    else:
                        predictions = predictions[:len(self.y_test)]
                        y_test = self.y_test
                else:
                    y_test = self.y_test
            elif model_name == 'LSTM':
                predictions = self.models[model_name].predict(self.X_test)
                y_test = self.y_test
            else:  # XGBoost
                X_test_2d = self.X_test.reshape(self.X_test.shape[0], -1)
                predictions = self.models[model_name].predict(X_test_2d)
                y_test = self.y_test
            
            predictions = np.array(predictions).flatten()
            y_test = np.array(y_test).flatten()
            
            min_len = min(len(predictions), len(y_test))
            predictions = predictions[:min_len]
            y_test = y_test[:min_len]
        
            # Calculating evaluation metrics
            mse = mean_squared_error(self.y_test, predictions)
            mae = mean_absolute_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            rmse = self._calculate_rmse(self.y_test, predictions)
            mape = self._calculate_mape(self.y_test, predictions)
            direction_acc = self._calculate_directional_accuracy(self.y_test, predictions)

            # Calculate returns and Sharpe Ratio
            returns = np.diff(predictions) / predictions[:-1]
            sharpe_ratio = self._calculate_sharpe_ratio(returns)

            self.results[model_name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'mape': mape,
                'directional_accuracy': direction_acc,
                'sharpe_ratio': sharpe_ratio,
                'predictions': predictions
            }

            return self.results[model_name]
        except Exception as e:
            print(f"Model {model_name} evaluation failed: {str(e)}")
            return None


    def compare_models(self):
        """Comparing the performance of all models"""
        comparison = {}
        for model_name in self.models.keys():
            if model_name in self.results:
                comparison[model_name] = {
                    'MSE': self.results[model_name]['mse'],
                    'MAE': self.results[model_name]['mae'],
                    'R2': self.results[model_name]['r2'],
                    'RMSE': self.results[model_name]['rmse'],
                    'MAPE': self.results[model_name]['mape'],
                    'Directional Accuracy': self.results[model_name]['directional_accuracy'],
                    'Sharpe Ratio': self.results[model_name]['sharpe_ratio']
                }
        return comparison

    def get_best_model(self):
        """Select the best model based on MSE"""
        best_model = min(self.results.items(),
                        key=lambda x: x[1]['mse'])
        return best_model[0], best_model[1]
    

# Use the previously cleaned data
framework = VIXPredictionFramework(cleaned_data1)

# preparing time series data
X_train, X_test, y_train, y_test = framework.prepare_time_series_data()

# train each model
framework.train_garch()
framework.train_arima()
framework.train_lstm()
framework.train_xgboost()

# Optimizing Hyperparameters for LSTM and XGBoost
lstm_params = framework.optimize_hyperparameters('LSTM')
xgb_params = framework.optimize_hyperparameters('XGBoost')

# Evaluate the performance of the models
for model_name in ['GARCH', 'ARIMA', 'LSTM', 'XGBoost']:
    framework.evaluate_model(model_name)

# Compare the performance of the models
comparison = framework.compare_models()
print("Model comparison results：")
print(comparison)

# The best model
best_model_name, best_model_results = framework.get_best_model()
print(f"The best model is: {best_model_name}")
print(f"Best model performance metrics: {best_model_results}")
