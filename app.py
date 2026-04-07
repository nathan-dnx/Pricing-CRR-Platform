"""
app.py — Interface Broker-Style
Plateforme de Pricing CRR & Delta-Hedging
Equipe 4302 — ESILV Fintech A4 — Mars 2026

Lancement : streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import os

from backend import (
    black_scholes_call,
    crr_call_price,
    crr_tree_nodes,
    monte_carlo_hedging,
    convergence_data,
    get_yfinance_data,
    simulate_martingale_single,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CRR Pricing Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGO SVG
# ─────────────────────────────────────────────────────────────────────────────
_logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
with open(_logo_path, "rb") as _f:
    _logo_b64 = base64.b64encode(_f.read()).decode()
LOGO_IMG = f'<img src="data:image/png;base64,{_logo_b64}" style="width:100%;max-width:260px;display:block;" alt="CRR Logo"/>'

_logo2_path = os.path.join(os.path.dirname(__file__), "logo2.png")
with open(_logo2_path, "rb") as _f:
    _logo2_b64 = base64.b64encode(_f.read()).decode()
LOGO2_IMG = f'<img src="data:image/png;base64,{_logo2_b64}" style="height:80px;display:block;" alt="CRR Logo"/>'

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE & THÈME
# ─────────────────────────────────────────────────────────────────────────────
CYAN    = "#00D4FF"
GREEN   = "#00FF88"
RED     = "#FF4466"
ORANGE  = "#FF9900"
PURPLE  = "#B388FF"
YELLOW  = "#FFD600"
BG_DARK = "#0A0E1A"
BG_CARD = "#111827"
BG_MID  = "#1A2235"
BORDER  = "#1E3A5F"
TEXT    = "#E2E8F0"
MUTED   = "#94A3B8"

# Styles de base partagés — sans legend/xaxis/yaxis (gérés par figure)
THEME_LAYOUT = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_DARK,
    font=dict(color=TEXT, family="monospace"),
    hoverlabel=dict(bgcolor=BG_MID, bordercolor=CYAN, font=dict(color=TEXT)),
)
AXIS_STYLE = dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS GLOBAL — BROKER STYLE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<style>
  /* ── Fond global */
  .stApp {{ background-color: {BG_DARK}; }}
  section[data-testid="stSidebar"] {{ background-color: #0D1321 !important; border-right: 1px solid {BORDER}; }}

  /* ── Sidebar header */
  .sidebar-logo {{
    font-family: monospace;
    font-size: 1.1rem;
    font-weight: 800;
    color: {CYAN};
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.5rem 0 0.2rem 0;
  }}
  .sidebar-sub {{
    font-size: 0.7rem;
    color: {MUTED};
    letter-spacing: 0.06em;
    margin-bottom: 1rem;
  }}

  /* ── Metric cards */
  .metric-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 1rem 1.2rem 0.8rem 1.2rem;
    position: relative;
    overflow: hidden;
  }}
  .metric-card::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }}
  .metric-card.cyan::before  {{ background: {CYAN}; }}
  .metric-card.green::before {{ background: {GREEN}; }}
  .metric-card.red::before   {{ background: {RED}; }}
  .metric-card.purple::before{{ background: {PURPLE}; }}

  .metric-label {{
    font-size: 0.68rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
    font-family: monospace;
  }}
  .metric-value {{
    font-size: 1.6rem;
    font-weight: 700;
    font-family: monospace;
    letter-spacing: 0.04em;
    line-height: 1.1;
  }}
  .metric-value.cyan   {{ color: {CYAN}; }}
  .metric-value.green  {{ color: {GREEN}; }}
  .metric-value.red    {{ color: {RED}; }}
  .metric-value.purple {{ color: {PURPLE}; }}

  .metric-sub {{
    font-size: 0.72rem;
    color: {MUTED};
    margin-top: 0.3rem;
    font-family: monospace;
  }}

  /* ── Header principal */
  .terminal-header {{
    background: linear-gradient(135deg, {BG_CARD} 0%, #0D1A2E 100%);
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .terminal-title {{
    font-family: monospace;
    font-size: 1.4rem;
    font-weight: 800;
    color: {CYAN};
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}
  .terminal-subtitle {{
    font-size: 0.72rem;
    color: {MUTED};
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
  }}
  .terminal-badge {{
    background: rgba(0, 212, 255, 0.12);
    border: 1px solid {CYAN};
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-family: monospace;
    font-size: 0.72rem;
    color: {CYAN};
    letter-spacing: 0.06em;
  }}

  /* ── Section titles */
  .section-title {{
    font-family: monospace;
    font-size: 0.8rem;
    font-weight: 700;
    color: {CYAN};
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-left: 3px solid {CYAN};
    padding-left: 0.7rem;
    margin: 1.2rem 0 0.8rem 0;
  }}

  /* ── Info band */
  .info-band {{
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-family: monospace;
    font-size: 0.78rem;
    color: {TEXT};
    margin: 0.5rem 0;
  }}
  .warn-band {{
    background: rgba(255, 153, 0, 0.08);
    border: 1px solid rgba(255, 153, 0, 0.3);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-family: monospace;
    font-size: 0.78rem;
    color: {ORANGE};
    margin: 0.5rem 0;
  }}
  .error-band {{
    background: rgba(255, 68, 102, 0.08);
    border: 1px solid rgba(255, 68, 102, 0.3);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-family: monospace;
    font-size: 0.78rem;
    color: {RED};
    margin: 0.5rem 0;
  }}

  /* ── Tabs */
  div[data-testid="stTabs"] button {{
    font-family: monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: {MUTED} !important;
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
  }}
  div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {CYAN} !important;
    border-bottom: 2px solid {CYAN} !important;
  }}
  div[data-testid="stTabContent"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 0 8px 8px 8px;
    padding: 1.2rem;
    margin-top: -1px;
  }}

  /* ── Sidebar sliders */
  .stSlider label {{ font-size: 0.75rem !important; color: {MUTED} !important; font-family: monospace !important; text-transform: uppercase; letter-spacing: 0.06em; }}
  .stSlider [data-testid="stTickBarMin"],
  .stSlider [data-testid="stTickBarMax"] {{ color: {MUTED} !important; font-size: 0.65rem !important; }}

  /* ── Number inputs */
  .stNumberInput label {{ font-size: 0.75rem !important; color: {MUTED} !important; font-family: monospace !important; text-transform: uppercase; letter-spacing: 0.06em; }}

  /* ── Divider */
  hr {{ border-color: {BORDER} !important; margin: 0.8rem 0; }}

  /* ── Hide streamlit default elements (pas le header : contient le toggle sidebar) */
  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
  header {{ background: transparent !important; }}

  /* ── Status dot */
  .status-dot {{
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: {GREEN};
    box-shadow: 0 0 6px {GREEN};
    margin-right: 6px;
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
  }}

  /* ── Table styling */
  .stDataFrame {{ border: 1px solid {BORDER} !important; border-radius: 6px; }}
  .stDataFrame th {{ background: {BG_MID} !important; color: {CYAN} !important; font-family: monospace !important; font-size: 0.72rem !important; text-transform: uppercase; }}
  .stDataFrame td {{ font-family: monospace !important; font-size: 0.78rem !important; color: {TEXT} !important; }}

  /* ── Button */
  .stButton button {{
    background: rgba(0, 212, 255, 0.1) !important;
    border: 1px solid {CYAN} !important;
    color: {CYAN} !important;
    font-family: monospace !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    border-radius: 6px !important;
    transition: all 0.2s;
  }}
  .stButton button:hover {{
    background: rgba(0, 212, 255, 0.2) !important;
    box-shadow: 0 0 12px rgba(0, 212, 255, 0.3) !important;
  }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cached_convergence(S0, K, T, r, sigma, N_max, prix_bs):
    return convergence_data(S0, K, T, r, sigma, N_max, prix_bs)

@st.cache_data(show_spinner=False)
def cached_yfinance(ticker):
    return get_yfinance_data(ticker)

@st.cache_data(show_spinner=False)
def cached_monte_carlo(S0, K, T, r, sigma, M, N_steps):
    return monte_carlo_hedging(S0, K, T, r, sigma, M, N_steps)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f'<div style="padding:0rem 0 2rem 0">{LOGO_IMG}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Pricing Engine v1.2 — Equipe 4302</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Mode
    st.markdown('<div class="section-title">Source</div>', unsafe_allow_html=True)
    mode = st.radio(
        "",
        options=["Manuel", "API (yfinance)"],
        label_visibility="collapsed",
    )

    # ── API mode
    if mode == "API (yfinance)":
        st.markdown('<div class="section-title">Actif</div>', unsafe_allow_html=True)
        TICKERS = {
            "Apple (AAPL)": "AAPL",
            "LVMH (MC.PA)": "MC.PA",
            "CAC 40 (^FCHI)": "^FCHI",
            "S&P 500 (^GSPC)": "^GSPC",
        }
        choix = st.selectbox("", list(TICKERS.keys()), label_visibility="collapsed")
        ticker = TICKERS[choix]

        with st.spinner("Connexion à Yahoo Finance..."):
            try:
                S0_api, sigma_api, _ = cached_yfinance(ticker)
                st.markdown(
                    f'<div class="info-band"><span class="status-dot"></span>'
                    f'<b>{choix}</b><br>'
                    f'S₀ = <b style="color:{CYAN}">{S0_api:.2f}</b> &nbsp;|&nbsp; '
                    f'σ = <b style="color:{CYAN}">{sigma_api:.2%}</b></div>',
                    unsafe_allow_html=True,
                )
                S0_input = S0_api
                sigma_input = sigma_api
            except Exception as e:
                st.markdown(f'<div class="error-band">⚠ Erreur API — passage en manuel<br><small>{e}</small></div>', unsafe_allow_html=True)
                S0_input = 100.0
                sigma_input = 0.20
    else:
        st.markdown('<div class="section-title">Spot & Volatilité</div>', unsafe_allow_html=True)
        S0_input = st.number_input("Spot S₀ (€)", min_value=1.0, value=100.0, step=1.0)
        sigma_input = st.slider("Volatilité σ", min_value=0.01, max_value=1.0, value=0.20, step=0.01, format="%.2f")

    st.markdown("---")
    st.markdown('<div class="section-title">Paramètres</div>', unsafe_allow_html=True)

    K       = st.number_input("Strike K (€)", min_value=1.0, value=float(round(S0_input)), step=1.0)
    T       = st.slider("Maturité T (ans)", min_value=0.1, max_value=3.0, value=1.0, step=0.1, format="%.1f")
    r       = st.slider("Taux r", min_value=0.0, max_value=0.15, value=0.05, step=0.005, format="%.3f")
    N       = st.slider("Pas N (CRR)", min_value=1, max_value=500, value=50, step=1)

    S0    = S0_input
    sigma = sigma_input

    st.markdown("---")

    # ── Moneyness badge
    ratio = S0 / K
    if ratio > 1.02:
        mny_color, mny_label = GREEN, "ITM"
    elif ratio < 0.98:
        mny_color, mny_label = RED, "OTM"
    else:
        mny_color, mny_label = YELLOW, "ATM"

    # ── Paramètres CRR affichés
    dt_disp = T / N
    u_disp  = np.exp(sigma * np.sqrt(dt_disp))
    d_disp  = 1 / u_disp
    p_disp  = (np.exp(r * dt_disp) - d_disp) / (u_disp - d_disp)

    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:0.5rem">
      <div class="metric-label">Paramètres CRR</div>
      <div style="font-family:monospace;font-size:0.8rem;line-height:1.8;color:{TEXT}">
        u = <span style="color:{CYAN}">{u_disp:.5f}</span><br>
        d = <span style="color:{CYAN}">{d_disp:.5f}</span><br>
        p = <span style="color:{CYAN}">{p_disp:.5f}</span><br>
        Δt = <span style="color:{MUTED}">{dt_disp:.5f}</span>
      </div>
      <div class="metric-sub">Moneyness :
        <span style="color:{mny_color};font-weight:700">{mny_label}</span>
        ({ratio:.3f})
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not (0 < p_disp < 1):
        st.markdown(
            f'<div class="error-band">⚠ Non-arbitrage violé<br>p = {p_disp:.4f} ∉ ]0,1[</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CALCULS EN TEMPS RÉEL
# ═══════════════════════════════════════════════════════════════════════════════

prix_bs, delta_bs = black_scholes_call(S0, K, T, r, sigma)
prix_crr          = crr_call_price(S0, K, T, r, sigma, N)
erreur            = abs(prix_crr - prix_bs)
erreur_pct        = (erreur / prix_bs * 100) if prix_bs > 1e-6 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="terminal-header">
  <div>
    {LOGO2_IMG}
    <div class="terminal-subtitle" style="margin-top:1.5rem">
      Cox-Ross-Rubinstein · Black-Scholes · Delta-Hedging Monte Carlo
      &nbsp;|&nbsp; ESILV Fintech A4 — Equipe 4302
    </div>
  </div>
  <div style="text-align:right">
    <div class="terminal-badge"><span class="status-dot"></span>LIVE · MODE {mode.upper()}</div>
    <div style="font-size:0.65rem;color:{MUTED};margin-top:0.3rem;font-family:monospace">
      S₀={S0:.2f} · K={K:.2f} · T={T:.1f}y · r={r:.3f} · σ={sigma:.2%} · N={N}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES PRINCIPALES — 4 CARDS
# ═══════════════════════════════════════════════════════════════════════════════

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card cyan">
      <div class="metric-label">Prix CRR</div>
      <div class="metric-value cyan">{prix_crr:.4f} €</div>
      <div class="metric-sub">Induction rétrograde · N={N} pas</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card green">
      <div class="metric-label">Prix Black-Scholes</div>
      <div class="metric-value green">{prix_bs:.4f} €</div>
      <div class="metric-sub">Formule fermée · référence continue</div>
    </div>""", unsafe_allow_html=True)

with c3:
    err_color = GREEN if erreur_pct < 0.5 else (ORANGE if erreur_pct < 2 else RED)
    st.markdown(f"""
    <div class="metric-card red">
      <div class="metric-label">Erreur |ε_N|</div>
      <div class="metric-value" style="color:{err_color}">{erreur:.4f} €</div>
      <div class="metric-sub">Relatif : {erreur_pct:.3f}% · O(1/N)</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card purple">
      <div class="metric-label">Delta BS (Δ)</div>
      <div class="metric-value purple">{delta_bs:.4f}</div>
      <div class="metric-sub">∂C/∂S = N(d₁) ∈ [0, 1]</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "◆ Convergence",
    "◆ Arbre Binomial",
    "◆ Stratégies Naïves",
    "◆ Delta-Hedging",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CONVERGENCE
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown('<div class="section-title">Convergence CRR → Black-Scholes</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-band">lim_{{N→∞}} C_CRR(N) = C_BS = '
        f'<b style="color:{CYAN}">{prix_bs:.4f} €</b> &nbsp;·&nbsp; '
        f'Erreur à N={N} : <b style="color:{err_color}">{erreur:.4f} €</b> '
        f'({erreur_pct:.3f}%)</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Calcul..."):
        Ns, prices, errors = cached_convergence(S0, K, T, r, sigma, N, prix_bs)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    # Courbe CRR
    fig.add_trace(go.Scatter(
        x=Ns, y=prices,
        mode="lines", name="C_CRR(N)",
        line=dict(color=CYAN, width=2),
        hovertemplate="N=%{x}<br>C_CRR=%{y:.4f} €<extra></extra>",
    ), row=1, col=1)

    # Ligne BS
    fig.add_trace(go.Scatter(
        x=[Ns[0], Ns[-1]], y=[prix_bs, prix_bs],
        mode="lines", name=f"C_BS = {prix_bs:.4f} €",
        line=dict(color=RED, width=1.5, dash="dash"),
        hovertemplate=f"C_BS = {prix_bs:.4f} €<extra></extra>",
    ), row=1, col=1)

    # Annotation point actuel N
    fig.add_trace(go.Scatter(
        x=[N], y=[prix_crr],
        mode="markers", name=f"N={N}",
        marker=dict(color=YELLOW, size=10, symbol="diamond",
                    line=dict(color="white", width=1.5)),
        hovertemplate=f"N={N}<br>C_CRR={prix_crr:.4f} €<extra></extra>",
    ), row=1, col=1)

    # Courbe erreur
    fig.add_trace(go.Scatter(
        x=Ns, y=errors,
        mode="lines", name="ε_N",
        line=dict(color=ORANGE, width=1.5),
        fill="tozeroy", fillcolor="rgba(255,153,0,0.08)",
        hovertemplate="N=%{x}<br>ε_N=%{y:.6f} €<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        **THEME_LAYOUT,
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(t=20, b=30, l=50, r=20),
    )
    fig.update_yaxes(title_text="Prix (€)", row=1, col=1, **AXIS_STYLE)
    fig.update_yaxes(title_text="ε_N (€)", type="log", row=2, col=1, **AXIS_STYLE)
    fig.update_xaxes(title_text="Nombre de pas N", row=2, col=1, **AXIS_STYLE)

    st.plotly_chart(fig, use_container_width=True)

    # Tableau comparatif
    key_ns = sorted(set(n for n in [5, 10, 25, 50, 100, 200, 500, N] if 1 <= n <= N))
    rows = []
    for n in key_ns:
        c = crr_call_price(S0, K, T, r, sigma, n)
        e = abs(c - prix_bs)
        rows.append({
            "N": n,
            "C_CRR(N)": f"{c:.6f} €",
            "C_BS": f"{prix_bs:.6f} €",
            "ε_N (€)": f"{e:.6f}",
            "ε_N (%)": f"{e/prix_bs*100:.4f}%" if prix_bs > 1e-6 else "—",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ARBRE BINOMIAL
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown('<div class="section-title">Arbre Binomial CRR</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 1])
    with col_r:
        N_tree = st.number_input("N (arbre)", 1, 6, min(N, 4), key="ntree",
                                  help="Max 6 pour la lisibilité")
        st.markdown(
            f'<div class="info-band" style="font-size:0.72rem">'
            f'u = {np.exp(sigma * np.sqrt(T/N_tree)):.4f}<br>'
            f'd = {np.exp(-sigma * np.sqrt(T/N_tree)):.4f}<br>'
            f'p = {((np.exp(r*T/N_tree) - np.exp(-sigma*np.sqrt(T/N_tree))) / (np.exp(sigma*np.sqrt(T/N_tree)) - np.exp(-sigma*np.sqrt(T/N_tree)))):.4f}'
            f'</div>',
            unsafe_allow_html=True
        )

    if N > 6:
        st.markdown(f'<div class="warn-band">⚠ N={N} — affichage limité à N_tree={N_tree}</div>', unsafe_allow_html=True)

    node_x, node_y, node_text, node_color, edge_x, edge_y, p_tree = crr_tree_nodes(
        S0, K, T, r, sigma, N_tree
    )

    fig_tree = go.Figure()

    # Arêtes
    fig_tree.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#1E3A5F", width=1.5),
        hoverinfo="none", showlegend=False,
    ))

    # Nœuds
    fig_tree.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=48,
            color=node_color,
            line=dict(color="white", width=1.5),
            opacity=0.9,
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(color="white", size=9, family="monospace"),
        hovertemplate="S = %{text} €<extra></extra>",
        showlegend=False,
    ))

    # Annotations p et 1-p sur la 1ère branche
    fig_tree.add_annotation(x=0.58, y=0.55, text=f"↑ p={p_tree:.3f}",
                             showarrow=False, font=dict(size=9, color=CYAN),
                             bgcolor="rgba(10,14,26,0.7)")
    fig_tree.add_annotation(x=0.58, y=-0.55, text=f"↓ 1-p={1-p_tree:.3f}",
                             showarrow=False, font=dict(size=9, color=MUTED),
                             bgcolor="rgba(10,14,26,0.7)")

    # Légende manuelle
    for color, label in [(GREEN, "ITM à maturité"), (MUTED, "OTM"), (CYAN, "Nœuds intermédiaires")]:
        fig_tree.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color), name=label,
        ))

    fig_tree.update_layout(
        **THEME_LAYOUT,
        title=dict(text=f"Arbre binomial — N={N_tree} pas", font=dict(color=CYAN, size=13)),
        height=420 + N_tree * 50,
        xaxis=dict(title="Etape i", tickvals=list(range(N_tree + 1)),
                   ticktext=[f"t={i}" for i in range(N_tree + 1)],
                   showgrid=False, zeroline=False, gridcolor=BORDER),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(t=50, b=30, l=30, r=30),
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # Tableau maturité
    dt_t = T / N_tree
    u_t = np.exp(sigma * np.sqrt(dt_t))
    d_t = 1.0 / u_t
    st.markdown(f'<div class="section-title">Nœuds à maturité (étape {N_tree})</div>', unsafe_allow_html=True)
    maturity_rows = []
    for j in range(N_tree + 1):
        s = S0 * (u_t ** (N_tree - j)) * (d_t ** j)
        payoff = max(s - K, 0.0)
        maturity_rows.append({
            "j": j,
            f"S_{{{N_tree},{j}}} (€)": f"{s:.4f}",
            "Payoff max(S−K,0) (€)": f"{payoff:.4f}",
            "Statut": "ITM ✅" if s > K else "OTM ❌",
        })
    st.dataframe(maturity_rows, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — STRATÉGIES NAÏVES
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown('<div class="section-title">Martingale — Doubling Strategy</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-band">La stratégie Martingale double la mise après chaque perte. '
        'Avec un capital fini, la <b>ruine est certaine</b> à long terme — '
        'contrairement au Delta-Hedging CRR (stratégie admissible et autofinancée).</div>',
        unsafe_allow_html=True,
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        W0_m   = st.number_input("Capital initial W₀ (€)", value=1000, step=100, key="w0m")
    with col_m2:
        mise_m = st.number_input("Mise de base u (€)", value=10, step=5, key="misem")
    with col_m3:
        tours  = st.slider("Nombre de tours", 10, 150, 60, key="toursm")

    if st.button("▶  SIMULER UNE TRAJECTOIRE"):
        hist, ruined = simulate_martingale_single(W0_m, mise_m, tours)

        fig_m = go.Figure()

        # Zone sous la courbe
        fill_color = "rgba(255,68,102,0.08)" if ruined else "rgba(0,255,136,0.06)"
        line_color = RED if ruined else GREEN

        fig_m.add_trace(go.Scatter(
            y=hist,
            mode="lines+markers",
            line=dict(color=line_color, width=2),
            marker=dict(size=5, color=line_color),
            fill="tozeroy", fillcolor=fill_color,
            name="Capital",
            hovertemplate="Tour %{x}<br>Capital=%{y:.0f} €<extra></extra>",
        ))

        # Ligne de ruine
        fig_m.add_hline(y=0, line_color=RED, line_width=2, line_dash="dash",
                        annotation_text="RUINE", annotation_font_color=RED,
                        annotation_position="top right")

        # Ligne capital initial
        fig_m.add_hline(y=W0_m, line_color=MUTED, line_width=1, line_dash="dot",
                        annotation_text=f"W₀={W0_m}€",
                        annotation_font_color=MUTED,
                        annotation_position="bottom right")

        fig_m.update_layout(
            **THEME_LAYOUT,
            title=dict(text="Evolution du capital — Martingale (Doubling Strategy)",
                       font=dict(color=CYAN, size=13)),
            xaxis=dict(title="Tours", **AXIS_STYLE),
            yaxis=dict(title="Capital (€)", **AXIS_STYLE),
            height=380,
            showlegend=False,
            margin=dict(t=50, b=40, l=60, r=20),
        )
        st.plotly_chart(fig_m, use_container_width=True)

        if ruined:
            st.markdown(
                f'<div class="error-band">💥 RUINE au tour {len(hist)-1} — '
                f'la mise exponentielle a dépassé le capital disponible.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="info-band">✅ Survie sur {len(hist)-1} tours — '
                f'Capital final : <b style="color:{GREEN}">{hist[-1]:.0f} €</b>. '
                f'Mais la ruine est inévitable à long terme.</div>',
                unsafe_allow_html=True,
            )



# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — DELTA-HEDGING MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown('<div class="section-title">Delta-Hedging — Simulation Monte Carlo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-band">Simulation de trajectoires GBM et calcul du P&L de couverture. '
        'Plus la fréquence de rebalancement augmente, plus le P&L converge vers 0.</div>',
        unsafe_allow_html=True,
    )

    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        M_sim     = st.slider("Trajectoires M", 10, 1000, 200, step=10, key="msim")
    with col_mc2:
        N_steps   = st.slider("Fréquence de rebalancement (pas)", 10, 252, 52, step=1, key="nsteps")

    if st.button("▶  LANCER LA SIMULATION MONTE CARLO"):
        with st.spinner("Simulation en cours..."):
            S_mc, pnl = cached_monte_carlo(S0, K, T, r, sigma, M_sim, N_steps)

        pnl_mean  = np.mean(pnl)
        pnl_std   = np.std(pnl)
        pnl_q5    = np.percentile(pnl, 5)
        pnl_q95   = np.percentile(pnl, 95)

        # ── Métriques MC
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.markdown(f"""<div class="metric-card cyan">
            <div class="metric-label">P&L Moyen</div>
            <div class="metric-value" style="color:{CYAN if abs(pnl_mean)<0.1 else RED};font-size:1.3rem">{pnl_mean:+.4f} €</div>
            <div class="metric-sub">Erreur de réplication moyenne</div></div>""", unsafe_allow_html=True)

        mc2.markdown(f"""<div class="metric-card purple">
            <div class="metric-label">Ecart-Type</div>
            <div class="metric-value purple" style="font-size:1.3rem">{pnl_std:.4f} €</div>
            <div class="metric-sub">Dispersion du P&L</div></div>""", unsafe_allow_html=True)

        mc3.markdown(f"""<div class="metric-card green">
            <div class="metric-label">Q5 / Q95</div>
            <div class="metric-value green" style="font-size:1.3rem">[{pnl_q5:.3f}, {pnl_q95:.3f}]</div>
            <div class="metric-sub">Intervalle de confiance 90%</div></div>""", unsafe_allow_html=True)

        mc4.markdown(f"""<div class="metric-card red">
            <div class="metric-label">Trajectoires</div>
            <div class="metric-value red" style="font-size:1.3rem">{M_sim}</div>
            <div class="metric-sub">{N_steps} rebalancements chacune</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)

        # ── Graphique 1 : Trajectoires GBM
        with col_g1:
            fig_traj = go.Figure()

            n_display = min(M_sim, 80)
            for i in range(n_display):
                fig_traj.add_trace(go.Scatter(
                    y=S_mc[i, :],
                    mode="lines",
                    line=dict(width=0.7, color="rgba(0,212,255,0.25)"),
                    showlegend=False, hoverinfo="skip",
                ))

            # Strike
            fig_traj.add_hline(
                y=K, line_color=RED, line_width=1.5, line_dash="dash",
                annotation_text=f"K={K:.0f}€", annotation_font_color=RED,
                annotation_position="top right",
            )

            fig_traj.update_layout(
                **THEME_LAYOUT,
                title=dict(text=f"Trajectoires GBM (affiché : {n_display}/{M_sim})",
                           font=dict(color=CYAN, size=12)),
                xaxis=dict(title="Pas (k)", **AXIS_STYLE),
                yaxis=dict(title="Prix S (€)", **AXIS_STYLE),
                height=360,
                margin=dict(t=50, b=40, l=60, r=20),
            )
            st.plotly_chart(fig_traj, use_container_width=True)

        # ── Graphique 2 : Histogramme P&L
        with col_g2:
            fig_pnl = go.Figure()

            fig_pnl.add_trace(go.Histogram(
                x=pnl,
                nbinsx=40,
                marker_color=CYAN,
                marker_line_color="rgba(0,212,255,0.4)",
                marker_line_width=0.5,
                opacity=0.8,
                name="P&L",
                hovertemplate="P&L=%{x:.3f} €<br>Freq=%{y}<extra></extra>",
            ))

            fig_pnl.add_vline(x=0, line_color=RED, line_width=1.5, line_dash="dash",
                              annotation_text="P&L=0", annotation_font_color=RED)
            fig_pnl.add_vline(x=pnl_mean, line_color=YELLOW, line_width=1, line_dash="dot",
                              annotation_text=f"μ={pnl_mean:.3f}", annotation_font_color=YELLOW)

            fig_pnl.update_layout(
                **THEME_LAYOUT,
                title=dict(text="Distribution du P&L de couverture",
                           font=dict(color=CYAN, size=12)),
                xaxis=dict(title="Erreur de réplication (€)", **AXIS_STYLE),
                yaxis=dict(title="Fréquence", **AXIS_STYLE),
                height=360,
                showlegend=False,
                margin=dict(t=50, b=40, l=60, r=20),
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        st.markdown(
            f'<div class="info-band">💡 Avec N_steps={N_steps} rebalancements — '
            f'P&L moyen = <b style="color:{CYAN}">{pnl_mean:+.4f} €</b>, '
            f'σ(P&L) = <b style="color:{PURPLE}">{pnl_std:.4f} €</b>. '
            f'Augmenter N_steps → σ(P&L) → 0 (couverture parfaite en continu).</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="info-band" style="text-align:center;padding:2rem">▶ Cliquez sur <b>LANCER LA SIMULATION</b> pour démarrer.</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    f'<div style="font-family:monospace;font-size:0.65rem;color:{MUTED};text-align:center">'
    f'CRR Pricing Terminal v1.2 &nbsp;·&nbsp; '
    f'Matthieu BALLISTE · Nathan DENOUX · Ilan CHADI · Ziad EL IDRISSI AMIRI &nbsp;·&nbsp; '
    f'Tuteur : Charaf LOUHMADI &nbsp;·&nbsp; ESILV Fintech A4 — Avril 2026'
    f'</div>',
    unsafe_allow_html=True,
)
