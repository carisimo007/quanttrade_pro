import os
import matplotlib.pyplot as plt
import pandas as pd
import base64

# ------------------------------------------------------------
# Función: _save_plot_price_with_signals
# Guarda un gráfico del precio con señales de Buy/Sell
# ------------------------------------------------------------
def _save_plot_price_with_signals(prices, signals, outpath):
    # Crear figura
    fig, ax = plt.subplots(figsize=(10,4))
    
    # Graficar los precios
    ax.plot(prices.index, prices.values, label='price')

    # Separar señales de compra y venta
    buys = signals[signals == 1].index
    sells = signals[signals == -1].index

    # Marcar señales con puntos
    ax.scatter(buys, prices.reindex(buys), marker='^', label='BUY', zorder=3)
    ax.scatter(sells, prices.reindex(sells), marker='v', label='SELL', zorder=3)

    ax.set_title("Price + Signals")
    ax.legend()

    # Guardar imagen en el archivo indicado
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------
# Función: _save_plot_nav
# Grafica la evolución del NAV (equity curve) del backtest
# ------------------------------------------------------------
def _save_plot_nav(nav_history, outpath):
    # Convertir la lista nav_history en DataFrame
    df = pd.DataFrame(nav_history, columns=["date", "nav"])

    # Asegurar que la columna 'date' sea datetime
    df['date'] = pd.to_datetime(df['date'])

    # Usar la fecha como índice
    df = df.set_index('date')

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df['nav'])
    ax.set_title("Equity Curve (NAV)")

    # Guardar imagen
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------
# Función: _save_plot_array
# Grafica múltiples caminos simulados (Monte Carlo u otros)
# ------------------------------------------------------------
def _save_plot_array(arr, outpath, title="Paths"):
    import numpy as np

    fig, ax = plt.subplots(figsize=(10,4))

    # Dibujar hasta 20 caminos
    n = min(arr.shape[0], 20)
    for i in range(n):
        ax.plot(arr[i, :], alpha=0.6)

    ax.set_title(title)

    # Guardar imagen
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------
# Función: _img_as_base64
# Convierte una imagen PNG a texto Base64 para incrustarla en HTML
# ------------------------------------------------------------
def _img_as_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode('utf-8')


# ------------------------------------------------------------
# Función principal: generate_dashboard
# Genera un dashboard HTML con:
# - Precio + señales
# - Equity curve
# - Caminos simulados (si hay)
# ------------------------------------------------------------
def generate_dashboard(prices, signals, backtest_result, paths=None,
                       out_html="results/dashboard.html", fig_dir="results/figures"):

    # Crear carpetas si no existen
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Rutas donde se guardarán las imágenes
    price_png = os.path.join(fig_dir, "price_signals.png")
    nav_png = os.path.join(fig_dir, "nav.png")
    paths_png = os.path.join(fig_dir, "paths.png")

    # Generar gráficos
    _save_plot_price_with_signals(prices, signals, price_png)
    _save_plot_nav(backtest_result.get("nav_history", []), nav_png)

    if paths is not None:
        _save_plot_array(paths, paths_png, title="Simulated paths")

    # Convertir imágenes a base64 para incrustar
    price_b64 = _img_as_base64(price_png)
    nav_b64 = _img_as_base64(nav_png)

    # HTML del dashboard
    body = f"""
    <html>
      <head><title>QuantTrade Pro - Dashboard</title></head>
      <body>
        <h1>QuantTrade Pro — Dashboard</h1>

        <h2>Price & Signals</h2>
        <img src="data:image/png;base64,{price_b64}" style="max-width:100%;height:auto"/>

        <h2>Equity Curve (valor total de la cuenta o de la cartera)</h2>
        <img src="data:image/png;base64,{nav_b64}" style="max-width:100%;height:auto"/>
    """

    # Agregar caminos simulados (si existen)
    if paths is not None:
        paths_b64 = _img_as_base64(paths_png)
        body += "<h2>Simulated Paths</h2>"
        body += f'<img src="data:image/png;base64,{paths_b64}" style="max-width:100%;height:auto"/>'

    body += "</body></html>"

    # Guardar HTML final
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(body)
