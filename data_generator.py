import numpy as np
import pandas as pd

def _make_dates(start_date="2022-01-01", days=252):
    return pd.bdate_range(start=start_date, periods=days)  # business days

def generate_gbm_series(S0=100.0, mu=0.0008, sigma=0.015, days=252, seed=None, start_date="2022-01-01"):
    """
    Genera una serie de precios siguiendo Geometric Brownian Motion.
    Devuelve pd.Series indexada por fechas.
    mu, sigma son en escala diaria.
    """
    if seed is not None:
        np.random.seed(seed)
    dt = 1.0  # 1 day
    eps = np.random.normal(loc=0.0, scale=1.0, size=days)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * eps
    log_price = np.cumsum(log_returns) + np.log(S0)
    prices = np.exp(log_price)
    dates = _make_dates(start_date=start_date, days=days)
    return pd.Series(prices, index=dates)

def generate_jump_diffusion_series(S0=100.0, mu=0.0008, sigma=0.015, lam=0.02, mu_j=-0.03, sigma_j=0.07, days=252, seed=None, start_date="2022-01-01"):
    """
    Merton's Jump Diffusion: GBM + Poisson jumps.
    lam = expected jumps per day
    mu_j, sigma_j = distribution of jump sizes (in log-return space)
    """
    if seed is not None:
        np.random.seed(seed)
    dt = 1.0
    prices = [S0]
    for _ in range(days-1):
        z = np.random.normal()
        jump_occurs = np.random.rand() < lam
        if jump_occurs:
            j = np.random.normal(loc=mu_j, scale=sigma_j)
        else:
            j = 0.0
        last = prices[-1]
        new = last * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z + j)
        prices.append(new)
    dates = _make_dates(start_date=start_date, days=days)
    return pd.Series(prices, index=dates)

def save_series_to_csv(series, start_date="2022-01-01", filename="data/sample.csv"):
    """
    Guarda la serie en CSV con columnas: date, price
    """
    df = series.reset_index()
    df.columns = ["date", "price"]
    df.to_csv(filename, index=False)
