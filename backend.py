"""
backend.py — Moteur Mathématique
Equipe 4302 — ESILV Fintech A4

Toute la logique de calcul, sans aucune dépendance à Streamlit ou Plotly.

Fonctions exportées :
    black_scholes_call(S0, K, T, r, sigma)      → (prix, delta)
    crr_call_price(S0, K, T, r, sigma, N)       → prix
    crr_tree_nodes(S0, K, T, r, sigma, N)       → (node_x, node_y, node_text, edge_x, edge_y)
    monte_carlo_hedging(S0, K, T, r, sigma, M, N_steps) → (S, pnl)
    get_yfinance_data(ticker)                   → (S0, sigma, history_df) ou raise
    convergence_data(S0, K, T, r, sigma, N_max, prix_bs) → (Ns, prices, errors)
"""

import numpy as np
import yfinance as yf
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes
# ─────────────────────────────────────────────────────────────────────────────

def black_scholes_call(S0: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """
    Formule fermée de Black-Scholes pour un call européen.
    Retourne (prix, delta_bs).
    """
    if T <= 0:
        prix = max(S0 - K, 0.0)
        delta = 1.0 if S0 > K else 0.0
        return prix, delta
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    prix = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return float(prix), float(delta)


# ─────────────────────────────────────────────────────────────────────────────
# CRR — prix
# ─────────────────────────────────────────────────────────────────────────────

def crr_call_price(S0: float, K: float, T: float, r: float, sigma: float, N: int) -> float:
    """
    Prix d'un call européen par induction rétrograde sur l'arbre binomial CRR.
    Complexité O(N²).
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)

    j = np.arange(N + 1)
    ST = S0 * (u ** (N - j)) * (d ** j)
    payoff = np.maximum(ST - K, 0.0)

    disc = np.exp(-r * dt)
    for _ in range(N):
        payoff = disc * (p * payoff[:-1] + (1 - p) * payoff[1:])

    return float(payoff[0])


# ─────────────────────────────────────────────────────────────────────────────
# CRR — données de l'arbre pour visualisation
# ─────────────────────────────────────────────────────────────────────────────

def crr_tree_nodes(
    S0: float, K: float, T: float, r: float, sigma: float, N: int
) -> tuple[list, list, list, list, list]:
    """
    Construit les coordonnées de l'arbre binomial pour N ≤ 100.

    Retourne :
        node_x, node_y   : coordonnées des nœuds (x=étape, y=rang symétrique)
        node_text        : label de chaque nœud (prix S_{i,j})
        node_color       : couleur selon ITM/OTM/intermédiaire
        edge_x, edge_y   : coordonnées des arêtes (avec None comme séparateur)
    """
    N = min(N, 100)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)

    node_x, node_y, node_text, node_color = [], [], [], []
    edge_x, edge_y = [], []

    for i in range(N + 1):
        for j in range(i + 1):
            S_ij = S0 * (u ** (i - j)) * (d ** j)
            nx, ny = float(i), float(i - 2 * j)

            node_x.append(nx)
            node_y.append(ny)
            node_text.append(f"{S_ij:.2f}")

            if i == N:
                node_color.append("#00FF88" if S_ij > K else "#666680")
            else:
                node_color.append("#00D4FF")

            if i < N:
                # Branche hausse
                edge_x += [nx, float(i + 1), None]
                edge_y += [ny, float(i + 1 - 2 * j), None]
                # Branche baisse
                edge_x += [nx, float(i + 1), None]
                edge_y += [ny, float(i + 1 - 2 * (j + 1)), None]

    return node_x, node_y, node_text, node_color, edge_x, edge_y, p


# ─────────────────────────────────────────────────────────────────────────────
# Delta-Hedging Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_hedging(
    S0: float, K: float, T: float, r: float, sigma: float,
    M_simulations: int, N_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simule M trajectoires GBM et calcule le P&L de couverture dynamique.

    S_{t+Δt} = S_t · exp[(r − σ²/2)·Δt + σ·√Δt·Z],  Z ~ N(0,1)

    Retourne :
        S   : (M, N_steps+1) — trajectoires du sous-jacent
        pnl : (M,)           — erreur de réplication à maturité
    """
    dt = T / N_steps
    S = np.zeros((M_simulations, N_steps + 1))
    S[:, 0] = S0
    delta = np.zeros((M_simulations, N_steps + 1))
    cash = np.zeros((M_simulations, N_steps + 1))

    # t=0 : on vend l'option, on achète delta_0 actions
    d1_init = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    prix_init = S0 * norm.cdf(d1_init) - K * np.exp(-r * T) * norm.cdf(d1_init - sigma * np.sqrt(T))
    delta[:, 0] = norm.cdf(d1_init)
    cash[:, 0] = prix_init - delta[:, 0] * S0

    np.random.seed(42)
    for k in range(1, N_steps + 1):
        T_restant = T - k * dt
        Z = np.random.standard_normal(M_simulations)
        S[:, k] = S[:, k - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        cash[:, k] = cash[:, k - 1] * np.exp(r * dt)

        if T_restant > 1e-9:
            d1 = (np.log(S[:, k] / K) + (r + 0.5 * sigma**2) * T_restant) / (sigma * np.sqrt(T_restant))
            delta[:, k] = norm.cdf(d1)
        else:
            delta[:, k] = np.where(S[:, k] > K, 1.0, 0.0)

        cash[:, k] -= (delta[:, k] - delta[:, k - 1]) * S[:, k]

    payoff = np.maximum(S[:, -1] - K, 0)
    pnl = cash[:, -1] + delta[:, -1] * S[:, -1] - payoff
    return S, pnl


# ─────────────────────────────────────────────────────────────────────────────
# Convergence CRR → BS
# ─────────────────────────────────────────────────────────────────────────────

def convergence_data(
    S0: float, K: float, T: float, r: float, sigma: float,
    N_max: int, prix_bs: float
) -> tuple[list[int], list[float], list[float]]:
    """
    Calcule les prix CRR pour N allant de 1 à N_max.
    Utilise un échantillonnage mixte dense/log pour N_max > 100.
    """
    if N_max <= 100:
        Ns = list(range(1, N_max + 1))
    else:
        dense = list(range(1, 51))
        sparse = list(dict.fromkeys(
            [int(round(x)) for x in np.geomspace(51, N_max, num=80)]
        ))
        Ns = sorted(set(dense + sparse + [N_max]))

    prices = [crr_call_price(S0, K, T, r, sigma, n) for n in Ns]
    errors = [abs(p - prix_bs) for p in prices]
    return Ns, prices, errors


# ─────────────────────────────────────────────────────────────────────────────
# API yfinance
# ─────────────────────────────────────────────────────────────────────────────

def get_yfinance_data(ticker: str) -> tuple[float, float, object]:
    """
    Récupère S0 (dernier cours) et sigma (volatilité historique annualisée) via yfinance.

    Retourne (S0, sigma, history_dataframe).
    Lève une exception si la récupération échoue.
    """
    data = yf.Ticker(ticker).history(period="1y")
    if data.empty:
        raise ValueError(f"Aucune donnée trouvée pour le ticker '{ticker}'.")
    S0 = float(data["Close"].iloc[-1])
    rendements = np.log(data["Close"] / data["Close"].shift(1))
    sigma = float(rendements.std() * np.sqrt(252))
    return S0, sigma, data


# ─────────────────────────────────────────────────────────────────────────────
# Martingale (Doubling Strategy) — une seule trajectoire
# ─────────────────────────────────────────────────────────────────────────────

def simulate_martingale_single(
    W0: float, mise_initiale: float, n_tours: int, seed: int = None
) -> tuple[list[float], bool]:
    """
    Simule UNE trajectoire de la stratégie Martingale (Doubling).

    Retourne (historique_capital, ruined).
    """
    if seed is not None:
        np.random.seed(seed)

    capital = W0
    mise = mise_initiale
    historique = [capital]
    ruined = False

    for _ in range(n_tours):
        if capital < mise:
            historique.append(0.0)
            ruined = True
            break
        if np.random.rand() > 0.5:
            capital += mise
            mise = mise_initiale
        else:
            capital -= mise
            mise = mise * 2
        historique.append(capital)

    return historique, ruined
