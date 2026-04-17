import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binom

st.set_page_config(page_title="Airline Overbooking Simulator", page_icon="✈️", layout="wide")

st.title("✈️ Airline Overbooking Revenue Simulator")
st.markdown("Monte Carlo simulation to find the optimal overbooking level.")

# --- Sidebar ---
st.sidebar.header("⚙️ Parameters")
CAPACITY     = st.sidebar.number_input("Plane Capacity (seats)", min_value=10, max_value=1000, value=100, step=10)
TICKET_PRICE = st.sidebar.number_input("Ticket Price ($)", min_value=10, max_value=5000, value=300, step=10)
VOUCHER_COST = st.sidebar.number_input("Bumping Voucher Cost ($)", min_value=10, max_value=5000, value=500, step=10)
NO_SHOW_PROB = st.sidebar.slider("No-Show Probability (%)", min_value=1, max_value=50, value=10) / 100
MAX_OVERBOOK = st.sidebar.slider("Max Overbooking Level to Test", min_value=5, max_value=50, value=25)
N_SIMS       = st.sidebar.select_slider("Monte Carlo Simulations", options=[10_000, 50_000, 100_000, 500_000], value=100_000)
SEED         = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42)

SHOW_PROB = 1 - NO_SHOW_PROB

# --- Simulation ---
@st.cache_data
def run_simulation(capacity, ticket_price, voucher_cost, show_prob, max_ob, n_sims, seed):
    rng = np.random.default_rng(seed=seed)
    results = []
    for ob in range(max_ob + 1):
        tickets_sold = capacity + ob
        show_ups     = rng.binomial(n=tickets_sold, p=show_prob, size=n_sims)
        revenue      = tickets_sold * ticket_price
        bumped       = np.maximum(show_ups - capacity, 0)
        profit       = revenue - bumped * voucher_cost
        results.append({
            "Overbook":    ob,
            "Sold":        tickets_sold,
            "Avg Profit":  profit.mean(),
            "Std":         profit.std(),
            "P(bump>0)":   (bumped > 0).mean(),
            "Avg Bumped":  bumped.mean(),
            "5% Worst":    np.percentile(profit, 5),
            "95th":        np.percentile(profit, 95),
        })
    return pd.DataFrame(results)

with st.spinner("Running Monte Carlo simulation..."):
    df = run_simulation(CAPACITY, TICKET_PRICE, VOUCHER_COST, SHOW_PROB, MAX_OVERBOOK, N_SIMS, SEED)

best_idx = df["Avg Profit"].idxmax()
best     = df.loc[best_idx]

# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Optimal Overbooking",       f"+{int(best['Overbook'])} seats")
col2.metric("Expected Profit (Optimal)", f"${best['Avg Profit']:,.2f}")
col3.metric("Baseline Profit (No OB)",   f"${df.loc[0,'Avg Profit']:,.2f}")
gain = best['Avg Profit'] - df.loc[0,'Avg Profit']
col4.metric("Gain from Overbooking",     f"${gain:,.2f}", delta=f"+{gain/df.loc[0,'Avg Profit']*100:.2f}%")

st.markdown("---")

# --- Dual Chart (matches notebook layout) ---
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("Expected Profit vs Overbooking Level",
                                    "Risk of Bumping vs Overbooking Level"))

# Left: profit + confidence band
fig.add_trace(go.Scatter(
    x=df["Overbook"], y=df["95th"],
    mode="lines", line=dict(width=0), showlegend=False,
    name="95th pct"), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df["Overbook"], y=df["5% Worst"],
    fill="tonexty", mode="lines", line=dict(width=0),
    fillcolor="rgba(31,119,180,0.2)", name="5th–95th percentile"), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df["Overbook"], y=df["Avg Profit"],
    mode="lines+markers", name="Expected profit",
    line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.add_vline(x=int(best["Overbook"]), line_dash="dash", line_color="red",
              annotation_text=f"Optimum = {int(best['Overbook'])}", row=1, col=1)

# Right: P(bump)
fig.add_trace(go.Scatter(
    x=df["Overbook"], y=df["P(bump>0)"],
    mode="lines+markers", name="P(bump > 0)",
    line=dict(color="darkorange", width=2),
    marker=dict(symbol="square")), row=1, col=2)
fig.add_vline(x=int(best["Overbook"]), line_dash="dash", line_color="red", row=1, col=2)

fig.update_xaxes(title_text="Seats overbooked", row=1, col=1)
fig.update_xaxes(title_text="Seats overbooked", row=1, col=2)
fig.update_yaxes(title_text="Profit ($)", row=1, col=1)
fig.update_yaxes(title_text="P(at least one passenger bumped)", tickformat=".0%", row=1, col=2)
fig.update_layout(height=440, template="plotly_white", legend=dict(orientation="h", y=-0.15))

st.plotly_chart(fig, use_container_width=True)

# --- Results Table ---
st.subheader("📋 Simulation Results Table")
display_df = df.copy()
display_df["P(bump>0)"] = display_df["P(bump>0)"].map(lambda x: f"{x:.2%}")
display_df["Avg Profit"] = display_df["Avg Profit"].map(lambda x: f"${x:,.2f}")
display_df["Std"]        = display_df["Std"].map(lambda x: f"{x:,.0f}")
display_df["Avg Bumped"] = display_df["Avg Bumped"].map(lambda x: f"{x:.3f}")
display_df["5% Worst"]   = display_df["5% Worst"].map(lambda x: f"${x:,.0f}")
display_df = display_df.drop(columns=["95th"])
display_df.columns = ["Overbook", "Sold", "Avg Profit", "Std", "P(bump>0)", "Avg Bumped", "5% Worst"]

optimal_ob = int(best["Overbook"])  # captured outside the function
def highlight_best(row):
    if row["Overbook"] == optimal_ob:
        return ["background-color: #c8f7c5;"] * len(row)
    return [""] * len(row)

st.dataframe(display_df.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)

# --- Marginal Analysis ---
st.subheader("📐 Marginal Analysis (Analytical)")
st.markdown("Add one more ticket only if **Marginal Revenue > Marginal Expected Cost**.")

marg_rows = []
for ob in range(MAX_OVERBOOK + 1):
    n_sold = CAPACITY + ob
    p_bump_given_show = 1 - binom.cdf(CAPACITY - 1, n_sold - 1, SHOW_PROB)
    marg_cost = VOUCHER_COST * SHOW_PROB * p_bump_given_show
    exceeds   = marg_cost > TICKET_PRICE
    marg_rows.append({
        "Extra Tickets": ob,
        "Marg Revenue ($)": TICKET_PRICE,
        "Marg Expected Cost ($)": round(marg_cost, 2),
        "Cost Exceeds Revenue?": "⚠️ Yes" if exceeds else "No",
    })

marg_df = pd.DataFrame(marg_rows)

def highlight_exceeds(row):
    if row["Cost Exceeds Revenue?"] == "Yes":
        return ["background-color: #ffe0e0;"] * len(row)
    return [""] * len(row)

st.dataframe(marg_df.style.apply(highlight_exceeds, axis=1), use_container_width=True, hide_index=True)

# --- Explanation ---
st.markdown("---")
st.subheader("📖 Model Details")
st.markdown(f"""
- **Revenue:** All tickets sold at ${TICKET_PRICE}/ticket — no refund for no-shows.
- **No-show model:** Each passenger independently shows up with probability **{SHOW_PROB:.0%}**, so show-ups ~ `Binomial(tickets_sold, {SHOW_PROB:.2f})`.
- **Bumping cost:** Every passenger beyond the **{CAPACITY}-seat** capacity receives a **${VOUCHER_COST} voucher**.
- **Optimal rule:** Overbook until the marginal expected voucher cost exceeds the ticket price (${TICKET_PRICE}).
- **Simulation:** {N_SIMS:,} Monte Carlo trials per overbooking level (seed = {SEED}).
""")
