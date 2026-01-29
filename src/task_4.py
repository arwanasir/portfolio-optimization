import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns


def optimize_portfolio(data, tsla_forecast_return):
    mu_historical = expected_returns.mean_historical_return(
        data[['BND', 'SPY']])
    mu = pd.Series({
        'TSLA': tsla_forecast_return,
        'BND': mu_historical['BND'],
        'SPY': mu_historical['SPY']
    })

    # 2. Compute Covariance Matrix (Risk Model)
    S = risk_models.sample_cov(data)

    # 3. Generate Key Portfolios
    # Max Sharpe Portfolio
    ef_sharpe = EfficientFrontier(mu, S)
    weights_sharpe = ef_sharpe.max_sharpe()
    perf_sharpe = ef_sharpe.portfolio_performance()

    # Min Volatility Portfolio
    ef_min_vol = EfficientFrontier(mu, S)
    weights_min_vol = ef_min_vol.min_volatility()
    perf_min_vol = ef_min_vol.portfolio_performance()

    return {
        "mu": mu, "S": S,
        "max_sharpe": {"weights": weights_sharpe, "perf": perf_sharpe},
        "min_vol": {"weights": weights_min_vol, "perf": perf_min_vol}
    }
