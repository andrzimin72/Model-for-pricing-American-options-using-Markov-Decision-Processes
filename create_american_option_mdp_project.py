#!/usr/bin/env python3
"""
Auto-generates the full 'american_option_mdp' project structure.
Run this script once to scaffold the entire package.
"""

import os
import sys

# Define project root name
PROJECT_NAME = "american_option_mdp"

# Create directory structure
dirs = [
    PROJECT_NAME,
    "tests"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    # Create __init__.py in each
    with open(os.path.join(d, "__init__.py"), "w") as f:
        pass  # empty file

print(f"✅ Created directories: {', '.join(dirs)}")

# File contents
FILES = {
    f"{PROJECT_NAME}/__init__.py": '''from .pricers import american_option_mdp, american_option_mdp_stoch_vol
from .utils import (
    compute_greeks_mdp,
    calibrate_implied_volatility,
    plot_exercise_boundary,
    plot_value_function,
    plot_policy_map,
    save_results_to_csv
)

__version__ = "1.0.0"
__all__ = [
    "american_option_mdp",
    "american_option_mdp_stoch_vol",
    "compute_greeks_mdp",
    "calibrate_implied_volatility",
    "plot_exercise_boundary",
    "plot_value_function",
    "plot_policy_map",
    "save_results_to_csv"
]
''',

    f"{PROJECT_NAME}/pricers.py": '''import numpy as np
from scipy.sparse import lil_matrix
from mdptoolbox.mdp import FiniteHorizon
from typing import Dict, Literal

def american_option_mdp(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    N_steps: int,
    option_type: Literal["put", "call"] = "put",
    S_min_factor: float = 0.2,
    S_max_factor: float = 3.0,
    N_states: int = 200,
) -> Dict:
    if option_type not in ("put", "call"):
        raise ValueError("option_type must be 'put' or 'call'")
    
    S_min = S0 * S_min_factor
    S_max = S0 * S_max_factor
    price_grid = np.linspace(S_min, S_max, N_states)
    s0_idx = np.argmin(np.abs(price_grid - S0))
    
    dt = T / N_steps
    discount = np.exp(-r * dt)
    mu = r - q
    
    dx = sigma * np.sqrt(3 * dt)
    u = np.exp(dx)
    d = np.exp(-dx)
    p_u = ((sigma**2 * dt + (mu * dt)**2) / (2 * dx**2)) + (mu * dt) / (2 * dx)
    p_d = ((sigma**2 * dt + (mu * dt)**2) / (2 * dx**2)) - (mu * dt) / (2 * dx)
    p_m = 1 - p_u - p_d
    
    if not (0 <= p_u <= 1 and 0 <= p_m <= 1 and 0 <= p_d <= 1):
        raise ValueError("Invalid trinomial probabilities — reduce dt.")
    
    A = 2
    P = [lil_matrix((N_states, N_states), dtype=np.float32) for _ in range(A)]
    
    for s in range(N_states):
        S = price_grid[s]
        Su, Sm, Sd = S * u, S, S * d
        iu = np.argmin(np.abs(price_grid - Su))
        im = np.argmin(np.abs(price_grid - Sm))
        id_ = np.argmin(np.abs(price_grid - Sd))
        P[0][s, iu] += p_u
        P[0][s, im] += p_m
        P[0][s, id_] += p_d
        if s == 0 or s == N_states - 1:
            P[0][s, :] = 0
            P[0][s, s] = 1.0
    P[1].setdiag(1.0)
    P = [p.tocsr() for p in P]
    
    R = [np.zeros(N_states, dtype=np.float32), np.zeros(N_states, dtype=np.float32)]
    if option_type == "call":
        R[1][:] = np.maximum(price_grid - K, 0.0)
    else:
        R[1][:] = np.maximum(K - price_grid, 0.0)
    
    fh = FiniteHorizon(P, R, discount, N_steps, skip_check=True)
    fh.run()
    
    return {
        "price": float(fh.V[s0_idx, 0]),
        "V": fh.V,
        "policy": fh.policy,
        "grid": price_grid,
        "dt": dt,
        "T": T,
        "type": option_type,
    }

def american_option_mdp_stoch_vol(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    N_steps: int,
    option_type: Literal["put", "call"] = "put",
    S_min_factor: float = 0.2,
    S_max_factor: float = 3.0,
    N_S: int = 50,
    v_min: float = 0.01,
    v_max: float = 0.5,
    N_v: int = 20,
) -> Dict:
    S_grid = np.linspace(S0 * S_min_factor, S0 * S_max_factor, N_S)
    v_grid = np.linspace(v_min, v_max, N_v)
    SS, VV = np.meshgrid(S_grid, v_grid, indexing='ij')
    states = np.stack([SS.ravel(), VV.ravel()], axis=1)
    N_states = len(states)
    s0_idx = np.argmin(np.linalg.norm(states - [S0, v0], axis=1))
    
    dt = T / N_steps
    discount = np.exp(-r * dt)
    A = 2
    P = [lil_matrix((N_states, N_states), dtype=np.float32) for _ in range(A)]
    
    for i, (S, v) in enumerate(states):
        if v <= 0: v = 1e-6
        mu_S = np.log(S) + (r - q - 0.5 * v) * dt
        sigma_S = np.sqrt(v * dt)
        mu_v = v + kappa * (theta - v) * dt
        sigma_v = xi * np.sqrt(v * dt)
        
        dS_vals = [mu_S - sigma_S, mu_S, mu_S + sigma_S]
        dv_vals = [mu_v - sigma_v, mu_v, mu_v + sigma_v]
        probs = np.full((3, 3), 1/9)
        
        for di in range(3):
            for dj in range(3):
                S_next = np.exp(dS_vals[di])
                v_next = np.clip(dv_vals[dj], v_min, v_max)
                j = np.argmin(np.linalg.norm(states - [S_next, v_next], axis=1))
                P[0][i, j] += probs[di, dj]
        row_sum = P[0][i, :].sum()
        if row_sum > 0:
            P[0][i, :] /= row_sum
        else:
            P[0][i, i] = 1.0
    P[1].setdiag(1.0)
    P = [p.tocsr() for p in P]
    
    R = [np.zeros(N_states, dtype=np.float32), np.zeros(N_states, dtype=np.float32)]
    payoff = np.maximum(states[:, 0] - K, 0.0) if option_type == "call" else np.maximum(K - states[:, 0], 0.0)
    R[1][:] = payoff
    
    fh = FiniteHorizon(P, R, discount, N_steps, skip_check=True)
    fh.run()
    
    return {
        "price": float(fh.V[s0_idx, 0]),
        "V": fh.V,
        "policy": fh.policy,
        "states": states,
        "S_grid": S_grid,
        "v_grid": v_grid,
        "type": option_type,
    }
''',

    f"{PROJECT_NAME}/utils.py": '''import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import root_scalar
from typing import Dict, Literal
from .pricers import american_option_mdp

def compute_greeks_mdp(mdp_result: Dict, S0: float, h: float = 0.01) -> Dict[str, float]:
    grid = mdp_result["grid"]
    V = mdp_result["V"]
    s0_idx = np.argmin(np.abs(grid - S0))
    
    if s0_idx == 0:
        delta = (V[s0_idx + 1, 0] - V[s0_idx, 0]) / (grid[s0_idx + 1] - grid[s0_idx])
    elif s0_idx == len(grid) - 1:
        delta = (V[s0_idx, 0] - V[s0_idx - 1, 0]) / (grid[s0_idx] - grid[s0_idx - 1])
    else:
        delta = (V[s0_idx + 1, 0] - V[s0_idx - 1, 0]) / (grid[s0_idx + 1] - grid[s0_idx - 1])
    
    if 0 < s0_idx < len(grid) - 1:
        gamma = (V[s0_idx + 1, 0] - 2 * V[s0_idx, 0] + V[s0_idx - 1, 0]) / ((grid[s0_idx + 1] - grid[s0_idx])**2)
    else:
        gamma = 0.0
    
    return {"delta": float(delta), "gamma": float(gamma), "price": float(V[s0_idx, 0])}

def calibrate_implied_volatility(
    market_price: float,
    S0: float, K: float, T: float, r: float, q: float,
    N_steps: int, option_type: Literal["put", "call"] = "put",
    sigma_guess: float = 0.2, tol: float = 1e-4
):
    def error_fn(sigma):
        try:
            res = american_option_mdp(S0, K, T, r, q, sigma, N_steps, option_type)
            return res["price"] - market_price
        except:
            return 1e6
    sol = root_scalar(error_fn, bracket=[0.01, 2.0], method="brentq", xtol=tol)
    return sol.root if sol.converged else sigma_guess

def plot_exercise_boundary(mdp_result, save_path: str = None):
    if "grid" not in mdp_result:
        print("Only 1D MDP supports boundary plot.")
        return
    grid = mdp_result["grid"]
    policy = mdp_result["policy"]
    T = mdp_result["T"]
    dt = mdp_result["dt"]
    opt_type = mdp_result["type"]
    
    boundary, times = [], []
    for t in range(policy.shape[1]):
        exercised = policy[:, t] == 1
        if np.any(exercised):
            idx = np.where(exercised)[0][-1] if opt_type == "put" else np.where(exercised)[0][0]
            boundary.append(grid[idx])
            times.append(T - t * dt)
    
    plt.figure(figsize=(8, 5))
    plt.plot(times, boundary, "o-", color="red")
    plt.xlabel("Time to Maturity")
    plt.ylabel("Stock Price")
    plt.title(f"Early Exercise Boundary ({opt_type.capitalize()} Option)")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_value_function(mdp_result, save_path: str = None):
    V = mdp_result["V"]
    grid = mdp_result["grid"]
    T = mdp_result["T"]
    time_axis = np.linspace(0, T, V.shape[1])
    S_mesh, T_mesh = np.meshgrid(time_axis, grid)
    plt.figure(figsize=(9, 6))
    plt.contourf(T_mesh, S_mesh, V, levels=30, cmap="viridis")
    plt.colorbar(label="Option Value")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Value Function $V(S, t)$")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_policy_map(mdp_result, save_path: str = None):
    policy = mdp_result["policy"]
    grid = mdp_result["grid"]
    T = mdp_result["T"]
    time_axis = np.linspace(0, T, policy.shape[1])
    S_mesh, T_mesh = np.meshgrid(time_axis, grid)
    plt.figure(figsize=(9, 6))
    plt.pcolormesh(T_mesh, S_mesh, policy, shading="auto", cmap="RdYlBu_r")
    plt.colorbar(label="Action (1=Exercise, 0=Continue)")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Optimal Policy Map")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def save_results_to_csv(results: dict, filename: str = None):
    if filename is None:
        from datetime import datetime
        filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame([results]).to_csv(filename, index=False)
    print(f"Saved to {filename}")
''',

    "tests/test_pricers.py": '''import pytest
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
''',

    "setup.py": '''from setuptools import setup, find_packages

setup(
    name="american_option_mdp",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "pymdptoolbox",
    ],
    python_requires=">=3.7",
)
''',

    "example.py": '''from american_option_mdp import (
    american_option_mdp,
    plot_exercise_boundary,
    plot_value_function,
    plot_policy_map,
    save_results_to_csv,
    compute_greeks_mdp
)

# Parameters
S0, K, T = 100, 100, 1.0
r, q, sigma = 0.05, 0.02, 0.2
N_steps = 50
option_type = "put"

# Price using MDP
result = american_option_mdp(
    S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
    N_steps=N_steps, option_type=option_type,
    N_states=200
)

print(f"American {option_type} price: ${result['price']:.4f}")

# Compute Greeks
greeks = compute_greeks_mdp(result, S0)
print(f"Delta: {greeks['delta']:.4f}, Gamma: {greeks['gamma']:.4f}")

# Save results
save_results_to_csv({
    "S0": S0, "K": K, "T": T, "r": r, "q": q, "sigma": sigma,
    "price": result["price"],
    "delta": greeks["delta"],
    "gamma": greeks["gamma"]
}, "results.csv")

# Generate plots (saved as PNG)
plot_exercise_boundary(result, "exercise_boundary.png")
plot_value_function(result, "value_function.png")
plot_policy_map(result, "policy_map.png")

print("\\n✅ All done! Check results.csv and .png files.")
'''
}

# Write all files
for filepath, content in FILES.items():
    with open(filepath, "w") as f:
        f.write(content)
    print(f"✅ Created {filepath}")

print("\n🎉 Project structure generated successfully!")
print(f"\nNext steps:")
print(f"1. Install: pip install -e .")
print(f"2. Run example: python example.py")
print(f"3. Run tests: pytest tests/ -v")