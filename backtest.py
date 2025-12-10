import pandas as pd

class BacktestEngine:
    """
    Motor de backtest sencillo:
    - signals: pd.Series aligned with price_series index, values {1, -1, 0}
    - reglas:
        * cuando signal == 1 y no está en posición -> comprar con todo el cash disponible (no leverage)
        * cuando signal == -1 y está en posición -> vender todo (ir a cash)
    - no shorting.
    - devuelve history (lista de eventos) y métricas.
    """

    def __init__(self, price_series, initial_cash=100000.0):
        self.price = price_series.copy().astype(float)
        self.initial_cash = float(initial_cash)

    def run_strategy(self, signals):
        # Asegurar alineamiento
        signals = signals.reindex(self.price.index).fillna(0).astype(int)
        cash = self.initial_cash
        position = 0  # número de acciones en cartera
        nav_history = []
        history = []

        for date, sig in signals.items():
            price = float(self.price.loc[date])
            # entrada larga
            if sig == 1 and position == 0:
                # comprar tantas acciones como permita el cash
                qty = int(cash // price)
                if qty > 0:
                    cost = qty * price
                    cash -= cost
                    position = qty
                    history.append([str(date.date()), "BUY", price, qty])
            # salida (cerrar posición)
            elif sig == -1 and position > 0:
                proceeds = position * price
                cash += proceeds
                history.append([str(date.date()), "SELL", price, position])
                position = 0
            nav = cash + position * price
            nav_history.append([str(date.date()), nav])

        final_nav = nav_history[-1][1] if nav_history else self.initial_cash
        pnl = final_nav - self.initial_cash

        result = {
            "initial_cash": self.initial_cash,
            "final_nav": final_nav,
            "pnl": pnl,
            "history": history,
            "nav_history": nav_history
        }
        return result
