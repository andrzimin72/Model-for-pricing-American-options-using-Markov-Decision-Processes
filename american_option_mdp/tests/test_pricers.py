import pytest
import numpy as np
from american_option_mdp.pricers import american_option_mdp

def binomial_american(S0, K, T, r, q, sigma, N, option_type='put'):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    ST = S0 * d**np.arange(N, -1, -1) * u**np.arange(0, N+1)
    V = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)
    for i in range(N-1, -1, -1):
        V = disc * (p * V[1:i+2] + (1-p) * V[0:i+1])
        S = S0 * d**np.arange(i, -1, -1) * u**np.arange(0, i+1)
        intrinsic = np.maximum(S - K, 0) if option_type == 'call' else np.maximum(K - S, 0)
        V = np.maximum(V, intrinsic)
    return V[0]

@pytest.mark.parametrize("otype", ["put", "call"])
def test_mdp_vs_binomial(otype):
    S0, K, T = 100, 100, 1.0
    r, q, sigma = 0.05, 0.02, 0.2
    N = 100
    mdp_price = american_option_mdp(S0, K, T, r, q, sigma, N, otype)["price"]
    binom_price = binomial_american(S0, K, T, r, q, sigma, N, otype)
    assert abs(mdp_price - binom_price) < 0.05