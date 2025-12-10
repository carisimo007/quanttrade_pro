import numpy as np

def eoq_trade_size(expected_volume, transaction_cost, holding_cost):
    """
    EOQ: q* = sqrt( (2 * D * S) / H )
    expected_volume: D (units per period)
    transaction_cost: S (cost per order/transaction)
    holding_cost: H (cost to hold one unit per period)
    """
    D = expected_volume
    S = transaction_cost
    H = holding_cost
    if H <= 0:
        raise ValueError("holding_cost must be > 0")
    q = np.sqrt((2.0 * D * S) / H)
    return float(q)
