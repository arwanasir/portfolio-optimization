import yfinance as yf
import pandas as pd
import os


def data_loader():
    tickers = ['TSLA', 'BND', 'SPY']
    start_date = '2015-01-01'
    end_date = '2026-01-15'

    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(
                "No data downloaded. Check your internet connection.")
        if 'Adj Close' in data.columns:
            df = data['Adj Close']
        elif 'Close' in data.columns:
            df = data['Close']
        else:
            df = data.xs('Close', axis=1, level=0)

        df = df.apply(pd.to_numeric, errors='coerce')
        df.index = pd.to_datetime(df.index)
        df = df.ffill().dropna()
        os.makedirs('../data/raw', exist_ok=True)
        df.to_csv('../data/raw/stock_data_cleaned.csv')

        print("Data loaded, cleaned, and saved successfully.")
        return df

    except Exception as e:
        print(f"Error during data loading: {e}")
        return None
