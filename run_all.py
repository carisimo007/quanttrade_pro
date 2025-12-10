import os
import json
import numpy as np
import pandas as pd

from data_generator import generate_gbm_series, generate_jump_diffusion_series, save_series_to_csv #crear datos y series sinteticas
from models_gbm import estimate_drift_volatility, simulate_gbm_paths # estimar parametros + simular caminos
from hmm_regime import fit_hmm_on_returns, regime_summary #modelos de regimen
from portfolio import optimize_min_variance, optimize_max_sharpe #markowitz
from eoq_trading import eoq_trade_size #eoq
from backtest import BacktestEngine #motor de backtest
from utils import load_series_from_csv, compute_returns 
from dashboard_generator import generate_dashboard #html con graficos

# definimos Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

def run_all(seed=None):
    """
    Ejecuta todo el pipeline:
    1) genera datos (GBM y Jump)
    2) estima parámetros
    3) ajusta HMM
    4) simula caminos Monte Carlo
    5) crea señales MA20 y backtest
    6) optimiza portafolio
    7) EOQ
    8) genera dashboard y guarda resultados
    """
    ensure_dirs()

    # Opción: semilla aleatoria o fija para reproducibilidad
    if seed is not None:
        np.random.seed(seed)
        print("Usando seed:", seed)
    else:
        print("Sin seed fija (saldrán datos distintos cada ejecución)")

    # 1) Generar datos sintéticos y guardarlos
    prices = generate_gbm_series(S0=100.0, mu=0.0008, sigma=0.015, days=252, seed=seed)
    save_series_to_csv(prices, start_date="2022-01-01", filename=os.path.join(DATA_DIR, "sample_gbm.csv"))

    prices_j = generate_jump_diffusion_series(S0=100.0, mu=0.0008, sigma=0.015, lam=0.02, mu_j=-0.03, sigma_j=0.07, days=252, seed=seed)
    save_series_to_csv(prices_j, start_date="2022-01-01", filename=os.path.join(DATA_DIR, "sample_jumpdiff.csv"))

    # 2) Cargar serie y estimar mu(drift: t.crecimiento) y sigma(volatilidad: variaciones)
    series = load_series_from_csv(os.path.join(DATA_DIR, "sample_gbm.csv"))
    mu_hat, sigma_hat = estimate_drift_volatility(series.values)
    print("Estimado mu, sigma:", mu_hat, sigma_hat)

    # 3) HMM sobre retornos
    returns = compute_returns(series)
    model, states = fit_hmm_on_returns(returns.values, n_states=2)
    summary = regime_summary(states)
    print("HMM summary:", summary)
    with open(os.path.join(RESULTS_DIR, "hmm_summary.json"), "w") as f:
        json.dump(summary, f)

    # 4) Simulación caminos futuros para Monte Carlo
    paths = simulate_gbm_paths(S0=series.iloc[0], mu=mu_hat, sigma=sigma_hat, days=252, n_paths=100, seed=seed)

    # Guardar ejemplo de caminos (dashboard se encargará de graficar)
    np.savez_compressed(os.path.join(RESULTS_DIR, "mc_paths.npz"), paths=paths)

    # 5) Señales simples (MA20) y Backtest
    df = series.to_frame(name='price')
    df['ma20'] = df['price'].rolling(20).mean()
    df['signal'] = 0
    df.loc[df['price'] > df['ma20'], 'signal'] = 1
    df.loc[df['price'] < df['ma20'], 'signal'] = -1
    signals = df['signal'].dropna()

    bt = BacktestEngine(series.loc[signals.index], initial_cash=100000.0)
    bt_result = bt.run_strategy(signals)

    # Guardar resumen del backtest
    with open(os.path.join(RESULTS_DIR, "backtest_summary.json"), "w") as f:
        json.dump({
            "initial": bt_result["initial_cash"],
            "final_nav": bt_result["final_nav"],
            "pnl": bt_result["pnl"]
        }, f)

    # Historial detallado
    hist_df = pd.DataFrame(bt_result["history"], columns=["date", "type", "price", "value"])
    hist_df.to_csv(os.path.join(RESULTS_DIR, "backtest_history.csv"), index=False)

    # 6) Optimización de portafolio (simulamos 3 activos)
    series_list = []
    for s in [1,2,3]:
        p = generate_gbm_series(S0=100*(1+s*0.1), mu=0.0005, sigma=0.02, days=252, seed=(None if seed is None else seed+s))
        series_list.append(p.values)

    P = np.vstack(series_list)
    rets = np.log(P[:,1:] / P[:,:-1])
    mean_returns = rets.mean(axis=1)
    cov = np.cov(rets)
    w_minvar, var_min = optimize_min_variance(mean_returns, cov) #optimiza riesgos
    w_sharpe, sharpe = optimize_max_sharpe(mean_returns, cov) #maximiza sharpe

    with open(os.path.join(RESULTS_DIR, "portfolio_weights.json"), "w") as f:
        json.dump({"minvar": w_minvar.tolist(), "sharpe": w_sharpe.tolist()}, f)

    # 7) EOQ
    q = eoq_trade_size(expected_volume=10000, transaction_cost=10, holding_cost=0.1)
    with open(os.path.join(RESULTS_DIR, "eoq_trade_size.json"), "w") as f:
        json.dump({"eoq_trade_size": q}, f)

    # 8) Generar dashboard (incluye figuras)
    generate_dashboard(prices=series, signals=signals, backtest_result=bt_result, paths=paths, out_html=os.path.join(RESULTS_DIR, "dashboard.html"), fig_dir=FIG_DIR)

    print("Ejecución completa. Revisá la carpeta 'results/' y 'results/figures'.")
    return True

if __name__ == "__main__":
    # Si querés reproducibilidad, pasá un número; sino deja None
    run_all(seed=None)
