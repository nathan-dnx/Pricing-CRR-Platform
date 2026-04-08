<p align="center">
  <img src="logo2.png" alt="CRR Pricing Platform" height="100"/>
</p>

**ESILV — Fintech A4 — Equipe 4302 — 2025/2026**

Interactive platform for pricing a European call option using two models — the Cox-Ross-Rubinstein binomial tree and the Black-Scholes closed-form formula — and comparing their outputs in real time.

---

## Motivation

The goal of this project is to compare two fundamentally different approaches to pricing the same financial instrument: a **European call option**.

- **Black-Scholes** assumes continuous time, continuous trading, and log-normally distributed returns. It yields an exact closed-form price.
- **CRR (Cox-Ross-Rubinstein)** works in discrete time by building a recombining binomial tree of possible asset prices. It is a constructive, numerical approach.

The key theoretical result that justifies the comparison is the convergence of CRR toward Black-Scholes as the number of steps N → ∞. This platform visualises that convergence and quantifies the error ε_N = |C_CRR(N) − C_BS|, which decreases at rate O(1/N).

---

## Mathematical Background

### The European Call Option

A European call option gives its holder the right (not the obligation) to buy an asset at price K (the strike) at maturity T. Its payoff at maturity is:

```
Payoff = max(S_T − K, 0)
```

where S_T is the asset price at time T.

### Black-Scholes Model

Under the Black-Scholes framework, the asset follows a Geometric Brownian Motion (GBM):

```
dS_t = r · S_t · dt + σ · S_t · dW_t
```

where r is the risk-free rate, σ is the volatility, and W_t is a standard Brownian motion.

The closed-form price of a European call is:

```
C_BS = S₀ · N(d₁) − K · e^(−rT) · N(d₂)

d₁ = [ ln(S₀/K) + (r + σ²/2) · T ] / (σ · √T)
d₂ = d₁ − σ · √T
```

where N(·) is the standard normal CDF.

The **Delta** of the option is:

```
Δ = ∂C/∂S = N(d₁) ∈ [0, 1]
```

It represents the sensitivity of the option price to a move in the underlying asset, and is the key quantity used in delta-hedging.

### Cox-Ross-Rubinstein (CRR) Model

CRR discretises time into N steps of length Δt = T/N. At each step, the asset either moves up by a factor u or down by a factor d:

```
u = e^(σ · √Δt)
d = 1/u
p = (e^(r·Δt) − d) / (u − d)        (risk-neutral probability)
```

The asset price at node (i, j) — step i, j down-moves — is:

```
S_{i,j} = S₀ · u^(i−j) · d^j
```

The option price is computed by **backward induction**: starting from the terminal payoffs at step N, the price is discounted back to t=0:

```
C_{i,j} = e^(−r·Δt) · [ p · C_{i+1,j} + (1−p) · C_{i+1,j+1} ]
```

### Convergence

As N increases, the CRR price converges to the Black-Scholes price:

```
lim_{N → ∞} C_CRR(N) = C_BS
```

The error decreases as O(1/N), with an oscillatory pattern (odd/even N) that is clearly visible in the platform's convergence tab.

### Delta-Hedging (Monte Carlo)

To validate the Black-Scholes model empirically, the platform simulates M asset trajectories under GBM and applies a **dynamic delta-hedging strategy**:

- At each rebalancing step k, the portfolio holds Δ_k shares of the underlying, financed by a cash account.
- At maturity, the replication error (P&L) measures how well the hedge tracked the option payoff.

Under perfect continuous hedging, the P&L should be zero. The Monte Carlo simulation shows the distribution of this error under discrete rebalancing.

---

## Platform Features

| Tab | Content |
|-----|---------|
| **Convergence** | CRR price vs N, convergence to C_BS, error ε_N on log scale |
| **Binomial Tree** | Interactive visualisation of the recombining tree (up to 100 steps) |
| **Naive Strategies** | Martingale (doubling) strategy simulation — illustration of ruin risk |
| **Delta-Hedging** | Monte Carlo GBM trajectories + P&L distribution of the hedge |

**Input modes:**
- **Manual** — set S₀, σ, K, T, r, N directly
- **API (yfinance)** — fetch live spot price and historical volatility for real tickers (AAPL, LVMH, CAC 40, S&P 500)

---

## Stack

- **Python 3.11+**
- [Streamlit](https://streamlit.io) — UI
- [NumPy](https://numpy.org) — numerical computations
- [SciPy](https://scipy.org) — normal CDF for Black-Scholes
- [Plotly](https://plotly.com/python/) — interactive charts
- [yfinance](https://github.com/ranaroussi/yfinance) — market data

---

## Run

```bash
streamlit run app.py
```
