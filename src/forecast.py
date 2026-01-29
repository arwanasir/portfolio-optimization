import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def generate_forecast_data(model_fit, steps=252):

    forecast_res = model_fit.get_forecast(steps=steps)
    forecast_values = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)

    return forecast_values, conf_int
