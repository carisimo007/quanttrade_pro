import pandas as pd
import numpy as np

def load_series_from_csv(path):
    """
    Lee CSV con columnas 'date' y 'price' y devuelve pd.Series indexada por fecha.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date")
    series = df['price'].sort_index()
    return series

def compute_returns(series):
    """
    Retornos logarítmicos (pd.Series).
    """
    return np.log(series).diff().dropna()
