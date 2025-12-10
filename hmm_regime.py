import numpy as np
from hmmlearn import hmm

def fit_hmm_on_returns(returns_array, n_states=2):
    """
    Ajusta Gaussian HMM a los retornos (1D).
    returns_array: 1D numpy array de log-returns
    Devuelve (model, states_array)
    """
    X = np.asarray(returns_array).reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
    model.fit(X)
    states = model.predict(X)
    return model, states

def regime_summary(states):
    """
    Resume duración y conteo de cada estado.
    states: array entero
    Devuelve dict con conteos y duración media.
    """
    out = {}
    unique = np.unique(states)
    out['total_days'] = int(len(states))
    for s in unique:
        idx = np.where(states == s)[0]
        out[f'state_{s}_days'] = int(len(idx))
    # duración media de regímenes (runs)
    runs = []
    current = states[0]
    length = 1
    for v in states[1:]:
        if v == current:
            length += 1
        else:
            runs.append((current, length))
            current = v
            length = 1
    runs.append((current, length))
    avg_lengths = {}
    for s in unique:
        lens = [l for (st, l) in runs if st == s]
        avg_lengths[f'state_{s}_avg_run'] = float(np.mean(lens)) if len(lens) > 0 else 0.0
    out.update(avg_lengths)
    return out
