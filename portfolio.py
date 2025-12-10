import numpy as np
from scipy.optimize import minimize

def optimize_min_variance(mean_returns, cov_matrix, allow_short=False):
    """
    Resuelve el problema de mínima varianza sujeto a sum(weights)=1
    mean_returns: vector (n,)
    cov_matrix: (n,n)
    allow_short: si True permite pesos negativos
    Devuelve (w, var)
    """
    n = len(mean_returns)
    x0 = np.ones(n) / n

    bounds = None if allow_short else [(0.0, 1.0) for _ in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    def var_fun(w):
        return float(w.T @ cov_matrix @ w)

    res = minimize(var_fun, x0=x0, constraints=constraints, bounds=bounds)
    w = res.x
    var = var_fun(w)
    return w, var

def optimize_max_sharpe(mean_returns, cov_matrix, rf=0.0, allow_short=False):
    """
    Maximiza el Sharpe ratio (media - rf) / std
    Devuelve (w, sharpe)
    """
    n = len(mean_returns)
    x0 = np.ones(n) / n
    bounds = None if allow_short else [(0.0, 1.0) for _ in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    def neg_sharpe(w):
        port_ret = float(w.T @ mean_returns)
        port_std = np.sqrt(float(w.T @ cov_matrix @ w))
        # evitar division por cero
        if port_std == 0:
            return 1e6
        return - (port_ret - rf) / port_std

    res = minimize(neg_sharpe, x0=x0, constraints=constraints, bounds=bounds)
    w = res.x
    sharpe = -neg_sharpe(w)
    return w, sharpe
