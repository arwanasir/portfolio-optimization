import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam


def split_data(df, test_size=0.2):
    """Splits data without shuffling to preserve time order."""
    train, test = train_test_split(df, test_size=test_size, shuffle=False)
    return train, test


def fit_optimized_arima(train_series, seasonal=False, m=1):
    print("Finding best ARIMA parameters...")
    model = auto_arima(train_series, seasonal=seasonal, m=m,
                       stepwise=True, suppress_warnings=True, trace=True)
    return model


def prepare_lstm_data(series, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    X = np.reshape(np.array(X), (len(X), window_size, 1))
    return X, np.array(y), scaler


def build_lstm(input_shape, neurons=50, layers=2, lr=0.001):
    model = Sequential()
    for i in range(layers):
        model.add(LSTM(units=neurons, return_sequences=(
            i < layers-1), input_shape=input_shape))
        model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model


def get_metrics(actual, predicted, model_name="Model"):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": f"{mape:.2f}%"}


"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam


def split_time_series_data(df, target_col, test_size=0.2):
    train, test = train_test_split(
        df,
        test_size=test_size,
        shuffle=False
    )

    return train, test


def plot_diagnostics(data, lags=40):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(data, lags=lags, ax=ax1)
    plot_pacf(data, lags=lags, ax=ax2)
    plt.tight_layout()
    return fig


def get_best_arima(train_series, seasonal=False, m=1):
    model = auto_arima(
        train_series,
        seasonal=seasonal,
        m=m,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=True
    )
    return model


def forecast_arima(model, periods):

    forecast, conf_int = model.predict(n_periods=periods, return_conf_int=True)
    return forecast, conf_int


def prepare_lstm_data(data, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def optimize_arima(train_series):
    model = auto_arima(
        train_series,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        trace=True
    )
    return model


def build_tuned_lstm(input_shape, neurons=50, layers=2, learning_rate=0.001):

    model = Sequential()
    for i in range(layers):
        is_last_lstm = (i == layers - 1)
        if i == 0:
            model.add(
                LSTM(units=neurons, return_sequences=not is_last_lstm, input_shape=input_shape))
        else:
            model.add(LSTM(units=neurons, return_sequences=not is_last_lstm))
        model.add(Dropout(0.2))

    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": f"{round(mape, 2)}%"
    }
    """
