import numpy as np
import pandas as pd

def estimate_drift_volatility(price_array):
    """
    price_array: 1D array-like de precios
    Devuelve (mu_hat, sigma_hat) en escala diaria basados en log-returns.
    """
    prices = np.asarray(price_array).astype(float).flatten()
    logp = np.log(prices)
    r = np.diff(logp)  # log-returns
    mu_hat = np.mean(r)
    sigma_hat = np.std(r, ddof=1)
    return mu_hat, sigma_hat

def simulate_gbm_paths(S0, mu, sigma, days=252, n_paths=100, seed=None):
    """
    Simula n_paths caminos GBM (incluye S0 como primer valor).
    Retorna array shape (n_paths, days)
    """
    if seed is not None:
        np.random.seed(seed)
    dt = 1.0
    paths = np.zeros((n_paths, days))
    for i in range(n_paths):
        eps = np.random.normal(size=days-1)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * eps
        log_price = np.concatenate([[np.log(S0)], np.cumsum(log_returns) + np.log(S0)])
        paths[i, :] = np.exp(log_price)
    return paths
