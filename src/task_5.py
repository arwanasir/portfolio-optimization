import pandas as pd
import numpy as np


def run_backtest(returns_df, weights, benchmark_col='SPY'):

    assets = ['TSLA', 'BND', 'SPY']
    portfolio_returns = returns_df[assets].dot(weights)
    cumulative_portfolio = (1 + portfolio_returns).cumprod()
    cumulative_benchmark = (1 + returns_df[benchmark_col]).cumprod()

    def calculate_stats(series):
        ann_return = series.mean() * 252
        ann_vol = series.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        return ann_return, ann_vol, sharpe

    p_stats = calculate_stats(portfolio_returns)
    b_stats = calculate_stats(returns_df[benchmark_col])

    covariance = portfolio_returns.cov(returns_df[benchmark_col])
    market_variance = returns_df[benchmark_col].var()
    beta = covariance / market_variance

    return {
        "cumulative_p": cumulative_portfolio,
        "cumulative_b": cumulative_benchmark,
        "metrics": {
            "Portfolio": [*p_stats, beta],
            "Benchmark": [*b_stats, 1.0]
        }
    }
