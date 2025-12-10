# QuantTrade Pro — Versión educativa y funcional

Proyecto didáctico que implementa:
- Generación de datos sintéticos (GBM y Jump Diffusion)
- Estimación de parámetros (drift y volatilidad)
- Detección de regímenes con HMM
- Simulación Monte Carlo de caminos
- Estrategia simple (MA20) y backtest
- Optimización de portafolio (mínima varianza / máximo Sharpe)
- EOQ aplicado a trade sizing
- Dashboard HTML con figuras y resultados
- Exportación de JSON y CSV

## Cómo ejecutar
1. Crear entorno e instalar dependencias:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

markdown
Copiar código

2. Ejecutar:
python run_all.py

markdown
Copiar código

Resultados en `results/` y figuras en `results/figures/`.

## Estructura
- `data_generator.py` — GBM, JumpDiffusion, guardado CSV
- `models_gbm.py` — estimación parámetros, simulación paths
- `hmm_regime.py` — ajuste HMM y resumen de regímenes
- `portfolio.py` — optimización Markowitz
- `eoq_trading.py` — EOQ
- `backtest.py` — motor de backtest simple
- `utils.py` — utilidades (I/O, retornos)
- `dashboard_generator.py` — genera PNGs y HTML
- `run_all.py` — orquesta todo

## Nota
Código educativo: pensado para ser claro y explicable en una presentación.