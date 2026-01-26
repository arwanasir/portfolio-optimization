# portfolio-optimization

## Project Overview

This project applies time series forecasting and Modern Portfolio Theory (MPT) to optimize investment portfolios for GMF Investments. The goal is to predict stock prices, analyze market trends, and construct optimal portfolios that balance risk and return.

## Assets Analyzed

- **TSLA** (Tesla Inc.) - High-growth stock
- **BND** (Vanguard Total Bond Market ETF) - Bond ETF for stability
- **SPY** (SPDR S&P 500 ETF) - Market index for diversification

## Project Structure

portfolio-optimization/
├── data/ # Raw and processed data
├── notebooks/ # Jupyter notebooks for analysis
├── src/ # Source code modules
├── scripts/ # Utility scripts
├── tests/ # Unit tests
└── docs/ # Documentation

## Key Tasks

1. **Data Extraction & EDA** - Fetch and analyze financial data
2. **Time Series Forecasting** - Build ARIMA/SARIMA and LSTM models
3. **Portfolio Optimization** - Generate Efficient Frontier
4. **Strategy Backtesting** - Validate portfolio performance

## Installation

```bash

git clone <repository-url>
cd portfolio-optimization

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```
