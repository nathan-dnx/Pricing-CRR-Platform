import streamlit as st
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import norm

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Pi² - Pricing CRR", layout="wide")

# ==========================================
# SIDEBAR : PANNEAU DE CONTRÔLE (INPUTS)
# ==========================================
st.sidebar.header("⚙️ Paramètres")

# Sélecteur de mode (Manuel / API)
mode = st.sidebar.radio("Mode d'alimentation", ["Manuel", "API (yfinance)"])

if mode == "API (yfinance)":
    # Menu déroulant pour choisir l'action proprement
    liste_tickers = {"Apple (AAPL)": "AAPL", "LVMH": "MC.PA", "CAC 40": "^FCHI", "S&P 500": "^GSPC"}
    choix = st.sidebar.selectbox("Choisissez l'actif", list(liste_tickers.keys()))
    ticker = liste_tickers[choix]
    
    with st.spinner('Connexion à Yahoo Finance...'):
        try:
            # Récupération des données via yfinance
            data = yf.Ticker(ticker).history(period="1y")
            
            # S0 : dernier cours de clôture
            S0 = float(data["Close"].iloc[-1])
            
            # Sigma : Volatilité historique annualisée
            rendements = np.log(data["Close"] / data["Close"].shift(1))
            sigma = float(rendements.std() * np.sqrt(252))
            
            st.sidebar.success(f"Données de {choix} récupérées !")
            st.sidebar.metric("Spot (S0) en direct", f"{S0:.2f}")
            st.sidebar.metric("Volatilité (σ) historique", f"{sigma:.2%}")
            
        except Exception as e:
            st.sidebar.error("Erreur API. Passage en mode manuel.")
            S0 = 100.0
            sigma = 0.20
else:
    # Mode Manuel : affichage des sliders pour S0 et sigma
    S0 = st.sidebar.number_input("Spot (S0)", min_value=1.0, value=100.0, step=1.0)
    sigma = st.sidebar.slider("Volatilité (σ)", min_value=0.01, max_value=1.0, value=0.20, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Autres Paramètres")

# Les autres variables restent manuelles dans les deux modes
K = st.sidebar.number_input("Strike (K)", min_value=1.0, value=S0, step=1.0) # Le strike s'adapte au S0
T = st.sidebar.slider("Maturité (T en années)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
r = st.sidebar.slider("Taux sans risque (r)", min_value=0.0, max_value=0.15, value=0.05, step=0.01)
N = st.sidebar.slider("Nombre de pas (N)", min_value=1, max_value=500, value=50, step=1)
# ==========================================
# MAIN PANEL : TABLEAU DE BORD
# ==========================================
st.title("📈 Projet Pi² : Plateforme de Pricing CRR")
st.markdown("*Modèle binomial Cox-Ross-Rubinstein vs Modèle continu Black-Scholes.*")

# --- MOTEUR MATHÉMATIQUE ---
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    prix = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return prix, delta

def crr_call_price(S0, K, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Valeurs finales du sous-jacent (étape N)
    ST = np.array([S0 * (u**(N-j)) * (d**j) for j in range(N+1)])
    payoff = np.maximum(ST - K, 0)
    
    # Induction rétrograde
    for i in range(N):
        payoff = np.exp(-r * dt) * (p * payoff[:-1] + (1 - p) * payoff[1:])
    return payoff[0]

def monte_carlo_hedging(S0, K, T, r, sigma, M_simulations, N_steps):
    dt = T / N_steps
    
    # Initialisation des matrices (M trajectoires, N étapes)
    S = np.zeros((M_simulations, N_steps + 1))
    S[:, 0] = S0
    delta = np.zeros((M_simulations, N_steps + 1))
    cash = np.zeros((M_simulations, N_steps + 1))
    
    # Prix théorique initial de l'option (prime reçue)
    d1_init = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    prix_bs_init = S0 * norm.cdf(d1_init) - K * np.exp(-r * T) * norm.cdf(d1_init - sigma * np.sqrt(T))
    
    # Étape 0 : On vend l'option (on reçoit le prix), on achète le Delta initial en actions
    delta[:, 0] = norm.cdf(d1_init)
    cash[:, 0] = prix_bs_init - delta[:, 0] * S0
    
    # Simulation des trajectoires et couverture dynamique
    for k in range(1, N_steps + 1):
        T_restant = T - k * dt
        
        # 1. Génération du mouvement brownien géométrique
        Z = np.random.standard_normal(M_simulations)
        S[:, k] = S[:, k-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        # Actualisation du cash avec le taux sans risque
        cash[:, k] = cash[:, k-1] * np.exp(r * dt)
        
        # 2. Calcul du nouveau Delta
        if T_restant > 0:
            d1 = (np.log(S[:, k] / K) + (r + 0.5 * sigma**2) * T_restant) / (sigma * np.sqrt(T_restant))
            delta[:, k] = norm.cdf(d1)
        else: # À maturité, le Delta vaut 1 si l'option est dans la monnaie, 0 sinon
            delta[:, k] = np.where(S[:, k] > K, 1.0, 0.0)
            
        # 3. Ajustement de la position (Achat/Vente d'actions)
        cash[:, k] -= (delta[:, k] - delta[:, k-1]) * S[:, k]
        
    # P&L Final (Erreur de couverture) à maturité
    payoff = np.maximum(S[:, -1] - K, 0)
    pnl = cash[:, -1] + delta[:, -1] * S[:, -1] - payoff
    
    return S, pnl

# --- CALCULS EN TEMPS RÉEL ---
prix_bs, delta_bs = black_scholes_call(S0, K, T, r, sigma)
prix_crr = crr_call_price(S0, K, T, r, sigma, N)
erreur = abs(prix_crr - prix_bs)

# --- AFFICHAGE DES MÉTRIQUES ---
st.subheader("Indicateurs Clés")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prix CRR ($C_{CRR}$)", f"{prix_crr:.4f} €")
col2.metric("Prix Black-Scholes ($C_{BS}$)", f"{prix_bs:.4f} €")
col3.metric("Erreur $|\epsilon_N|$", f"{erreur:.4f} €")
col4.metric("Delta ($\Delta_{BS}$)", f"{delta_bs:.4f}")

st.markdown("---")

# ==========================================
# LES ONGLETS D'ANALYSE (VISUALISATION)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📉 Convergence CRR/BS", "🌳 Arbre Binomial", "🎲 Stratégies Naïves", "🛡️ Delta-Hedging (Monte Carlo)"])

with tab1:
    st.header("Convergence vers Black-Scholes")
    st.markdown("Évolution du prix CRR en fonction du nombre de pas $N$ par rapport à la valeur théorique continue.")
    
    with st.spinner("Calcul de la convergence en cours..."):
        # On calcule les prix CRR pour différentes valeurs de N (de 1 à max_N)
        max_steps = max(N, 50) # On va jusqu'au N choisi, ou au moins 50 pour voir la courbe
        plage_N = list(range(1, max_steps + 1))
        prix_crr_liste = [crr_call_price(S0, K, T, r, sigma, n) for n in plage_N]

        # Création du graphique Plotly
        fig_conv = go.Figure()
        
        # Courbe CRR (en dents de scie qui s'aplatit)
        fig_conv.add_trace(go.Scatter(x=plage_N, y=prix_crr_liste, mode='lines', name='Prix CRR', line=dict(color='#1f77b4')))
        
        # Ligne droite Black-Scholes (Asymptote)
        fig_conv.add_trace(go.Scatter(x=[1, max_steps], y=[prix_bs, prix_bs], mode='lines', name='Prix Black-Scholes', line=dict(color='red', dash='dash')))

        fig_conv.update_layout(
            title="Convergence asymptotique du modèle binomial",
            xaxis_title="Nombre de pas (N)",
            yaxis_title="Prix du Call",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_conv, use_container_width=True)

with tab2:
    st.header("Visualisation de l'Arbre CRR")
    
    if N > 6:
        st.warning(f"⚠️ Le nombre de pas actuel est N = {N}. L'arbre est trop dense pour être affiché lisiblement. Veuillez régler N sur une valeur inférieure ou égale à 6 dans la barre latérale.")
    else:
        st.markdown(f"Trajectoires possibles du sous-jacent pour $N={N}$ pas.")
        
        # Recalcul des paramètres de l'arbre
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        # Construction des coordonnées des nœuds et des arêtes
        node_x, node_y, node_text = [], [], []
        edge_x, edge_y = [], []
        
        for i in range(N + 1):
            for j in range(i + 1):
                S_ij = S0 * (u**(i-j)) * (d**j)
                node_x.append(i)
                node_y.append(S_ij)
                node_text.append(f"{S_ij:.2f}")
                
                if i < N: # Si on n'est pas à la dernière étape, on crée les branches vers i+1
                    S_up = S_ij * u
                    S_down = S_ij * d
                    # Branche Haut
                    edge_x.extend([i, i+1, None])
                    edge_y.extend([S_ij, S_up, None])
                    # Branche Bas
                    edge_x.extend([i, i+1, None])
                    edge_y.extend([S_ij, S_down, None])

        # Dessin de l'arbre
        fig_tree = go.Figure()
        
        # Ajout des branches (lignes grises)
        fig_tree.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='lightgray', width=1.5), hoverinfo='none', showlegend=False))
        
        # Ajout des nœuds (points bleus avec le prix écrit au-dessus)
        fig_tree.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center", 
                                      marker=dict(size=12, color='#00CC96', line=dict(width=2, color='DarkSlateGrey')), showlegend=False))
        
        fig_tree.update_layout(
            xaxis_title="Pas de temps (i)",
            yaxis_title="Prix du sous-jacent (S)",
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, dtick=1),
            yaxis=dict(showgrid=False, zeroline=False)
        )
        st.plotly_chart(fig_tree, use_container_width=True)

with tab3:
    st.header("Analyse de la Martingale (Doubling Strategy)")
    st.markdown("Contrairement au Delta-Hedging du modèle CRR qui neutralise le risque, la stratégie naïve de la Martingale espère un retournement en doublant la mise à chaque perte. Cette simulation illustre le risque de ruine certaine avec un capital fini.")
    
    # Paramètres de la simulation
    col_m1, col_m2, col_m3 = st.columns(3)
    W0 = col_m1.number_input("Capital Initial (€)", value=1000, step=100)
    mise_initiale = col_m2.number_input("Mise de base (€)", value=10, step=5)
    n_tours = col_m3.slider("Nombre de paris max", 10, 150, 50)
    
    # Bouton pour lancer la simulation (très stylé en présentation)
    if st.button("🎲 Lancer une simulation (Monte Carlo)"):
        capital = W0
        mise = mise_initiale
        historique_capital = [capital]
        
        for i in range(n_tours):
            if capital < mise: 
                # Ruine : le joueur n'a plus assez d'argent pour suivre la mise exponentielle
                historique_capital.append(0)
                break
                
            # Simulation d'un pari type "Pile ou Face" (Marché symétrique, p=0.5)
            if np.random.rand() > 0.5: # Victoire
                capital += mise
                mise = mise_initiale # On réinitialise la mise à 10€
            else: # Défaite
                capital -= mise
                mise *= 2 # Doubling strategy : on double la mise
                
            historique_capital.append(capital)
            
        # Création du graphique Plotly pour voir la chute
        fig_martingale = go.Figure()
        
        # Ligne d'évolution du capital
        fig_martingale.add_trace(go.Scatter(
            y=historique_capital, 
            mode='lines+markers', 
            name="Capital du Joueur", 
            line=dict(color='orange', width=2),
            marker=dict(size=6)
        ))
        
        # Ligne de Ruine (Rouge)
        fig_martingale.add_trace(go.Scatter(
            x=[0, len(historique_capital)-1], 
            y=[0, 0], 
            mode='lines', 
            name="Seuil de Ruine", 
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig_martingale.update_layout(
            title="Évolution du capital et Risque de Ruine",
            xaxis_title="Nombre de paris (t)",
            yaxis_title="Capital restant (€)",
            template="plotly_white"
        )
        st.plotly_chart(fig_martingale, use_container_width=True)
        
        # Affichage du résultat final
        if historique_capital[-1] == 0:
            st.error(f"💥 RUINE TOTALE au tour {len(historique_capital)-1} ! Le capital n'était pas suffisant pour couvrir la mise exponentielle.")
        else:
            st.success(f"✅ Survie réussie sur {n_tours} tours. Capital final : {historique_capital[-1]} €. Mais jusqu'à quand ?")

with tab4:
    st.header("Couverture Dynamique (Delta-Hedging)")
    st.markdown("Simulation de trajectoires par mouvement brownien géométrique et calcul de l'erreur de réplication (PnL) à maturité.")
    
    col_mc1, col_mc2 = st.columns(2)
    M_sim = col_mc1.slider("Nombre de trajectoires (M)", min_value=10, max_value=1000, value=100, step=10)
    N_steps_mc = col_mc2.slider("Fréquence de rebalancement (Pas)", min_value=10, max_value=252, value=50, step=10)
    
    if st.button("🚀 Lancer le Delta-Hedging"):
        with st.spinner("Simulation Monte Carlo en cours..."):
            trajectoires, pnl = monte_carlo_hedging(S0, K, T, r, sigma, M_sim, N_steps_mc)
            
            col_res1, col_res2 = st.columns(2)
            
            # Graphique 1 : Trajectoires du sous-jacent
            fig_traj = go.Figure()
            # On affiche max 100 trajectoires pour ne pas faire lagger le navigateur
            for i in range(min(M_sim, 100)):
                fig_traj.add_trace(go.Scatter(y=trajectoires[i, :], mode='lines', line=dict(width=1, color='lightblue'), opacity=0.5, showlegend=False))
            fig_traj.add_trace(go.Scatter(y=[K]*(N_steps_mc+1), mode='lines', line=dict(color='red', dash='dash', width=2), name="Strike (K)"))
            
            fig_traj.update_layout(title="Trajectoires simulées du sous-jacent", xaxis_title="Pas de temps (k)", yaxis_title="Prix (S)", template="plotly_white")
            col_res1.plotly_chart(fig_traj, width='stretch')
            
            # Graphique 2 : Histogramme du P&L (Erreur de couverture)
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Histogram(x=pnl, nbinsx=30, marker_color='#00CC96'))
            fig_pnl.update_layout(title="Distribution du P&L de Couverture", xaxis_title="Erreur de réplication (€)", yaxis_title="Fréquence", template="plotly_white")
            
            # Ligne verticale à 0
            fig_pnl.add_vline(x=0, line_dash="dash", line_color="red")
            col_res2.plotly_chart(fig_pnl, width='stretch')
            
            st.success(f"Erreur moyenne de couverture (PnL) : {np.mean(pnl):.4f} € | Écart-type : {np.std(pnl):.4f} €")
            st.info("💡 Plus la fréquence de rebalancement augmente, plus la distribution du P&L se resserre autour de zéro (couverture parfaite en continu).")