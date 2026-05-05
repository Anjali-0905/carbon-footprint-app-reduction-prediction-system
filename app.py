"""
Carbon Footprint Prediction & Reduction System
===============================================
Run:
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn
    streamlit run app.py

Place  carbon_model.pkl  and  energy_emissions_semireal.csv  in the SAME
folder as this file.  The app auto-detects them.
"""

from __future__ import annotations   # fixes list[dict] on Python 3.9

import os, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # headless — must come before pyplot
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Carbon Footprint System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:700;color:#1a6b3c;text-align:center;margin-bottom:.2rem}
.sub-header{font-size:1rem;color:#555;text-align:center;margin-bottom:1rem}
.suggestion-box{background:#f1f8e9;border-radius:10px;padding:.9rem;
    margin:.4rem 0;border:1px solid #aed581}
.alert-high{background:#ffebee;border-left:5px solid #c62828;padding:.8rem;border-radius:8px}
.alert-med {background:#fff3e0;border-left:5px solid #e65100;padding:.8rem;border-radius:8px}
.alert-low {background:#e8f5e9;border-left:5px solid #2e7d32;padding:.8rem;border-radius:8px}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Coal_Consumption","Gas_Consumption","Oil_Consumption",
    "Solar_Generation","Wind_Generation","Hydro_Generation",
    "Energy_Demand","Population",
]
TARGET_COL = "CO2_Emission"
PALETTE    = ["#2e7d32","#43a047","#66bb6a","#a5d6a7","#1b5e20","#4caf50"]

# ─────────────────────────────────────────────────────────────
# AUTO-LOCATE FILES
# ─────────────────────────────────────────────────────────────
APP_DIR      = os.path.dirname(os.path.abspath(__file__))
CSV_DEFAULT  = os.path.join(APP_DIR, "energy_emissions_semireal.csv")
PKL_DEFAULT  = os.path.join(APP_DIR, "carbon_model.pkl")


# ═══════════════════════════════════════════════════════════════
# MODULE 1 — DATA
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ═══════════════════════════════════════════════════════════════
# MODULE 2 — ML
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model(pkl_path: str, csv_path: str):
    if os.path.exists(pkl_path):
        try:
            import tensorflow  # noqa
            with open(pkl_path, "rb") as f:
                m = pickle.load(f)
            return m, "Keras (your model)"
        except Exception:
            pass

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    df = pd.read_csv(csv_path)
    X  = df[FEATURE_COLS].values
    y  = df[TARGET_COL].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr",    GradientBoostingRegressor(n_estimators=300,
                                              learning_rate=0.05,
                                              max_depth=3,
                                              random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe, "Scikit-learn GBR (fallback)"


def predict_one(model, backend: str, features: np.ndarray) -> float:
    x = features.reshape(1, -1)
    if "Keras" in backend:
        return float(np.squeeze(model.predict(x, verbose=0)))
    return float(model.predict(x)[0])


# ═══════════════════════════════════════════════════════════════
# MODULE 3 — FORECAST
# ═══════════════════════════════════════════════════════════════
def forecast_state(df: pd.DataFrame, state: str, years_ahead: int,
                   model, backend: str) -> pd.DataFrame:
    sdf       = df[df["State"] == state].sort_values("Year")
    last_year = int(sdf["Year"].max())
    rows = []
    for yr in range(last_year + 1, last_year + years_ahead + 1):
        row = {"Year": yr, "State": state}
        for col in FEATURE_COLS:
            coeff = np.polyfit(sdf["Year"], sdf[col], 1)
            row[col] = max(0.0, float(np.polyval(coeff, yr)))
        rows.append(row)
    fdf   = pd.DataFrame(rows)
    preds = [predict_one(model, backend,
                         fdf[FEATURE_COLS].values[i]) for i in range(len(fdf))]
    fdf[TARGET_COL] = preds
    return fdf


# ═══════════════════════════════════════════════════════════════
# MODULE 4 — REDUCTION SUGGESTIONS
# ═══════════════════════════════════════════════════════════════
def get_suggestions(row: dict, avg_em: float) -> list:
    sugs = []
    em      = row.get("CO2_Emission", avg_em)
    coal    = row.get("Coal_Consumption", 0)
    solar   = row.get("Solar_Generation", 0)
    wind    = row.get("Wind_Generation", 0)
    hydro   = row.get("Hydro_Generation", 0)
    dem     = row.get("Energy_Demand", 1) or 1
    ren_pct = (solar + wind + hydro) / dem * 100

    if coal > 150:
        sugs.append({"cat":"⚡ Energy Source","pri":"High",
            "text":"Shift 20% coal load to gas or renewables.",
            "impact":f"~{coal*0.2*2.1:.0f} MT CO₂ saved/yr"})
    if coal > 80:
        sugs.append({"cat":"⚡ Energy Source","pri":"Medium",
            "text":"Install carbon capture on coal plants (CCS).",
            "impact":"Up to 90% point-source capture"})
    if ren_pct < 30:
        sugs.append({"cat":"☀️ Renewables","pri":"High",
            "text":f"Renewable share only {ren_pct:.1f}% — scale to 40%+.",
            "impact":"Significant grid emission factor reduction"})
    if solar < 60:
        sugs.append({"cat":"☀️ Renewables","pri":"Medium",
            "text":"Launch rooftop solar subsidy; target +30 BU.",
            "impact":"~15–20 MT CO₂ avoided/yr"})
    if dem > 500:
        sugs.append({"cat":"🏭 Efficiency","pri":"High",
            "text":"Mandatory industrial energy audits; 10% demand cut.",
            "impact":f"~{dem*0.1*0.82:.0f} MT CO₂ avoided"})
    sugs.append({"cat":"🏭 Efficiency","pri":"Low",
        "text":"Fast-track BEE star-ratings & full LED switchover.",
        "impact":"5–8% electricity savings"})
    if em > avg_em * 1.2:
        sugs.append({"cat":"🌲 Offsets","pri":"High",
            "text":"Emission 20%+ above avg — invest in afforestation.",
            "impact":"~1 MT CO₂/yr per 800 ha"})
    sugs.append({"cat":"🚌 Transport","pri":"Medium",
        "text":"EV adoption push with charging-infra subsidies.",
        "impact":"Road transport ~30% of urban emissions"})
    return sugs


# ═══════════════════════════════════════════════════════════════
# MODULE 5 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════
def fig_trend(df, state):
    sdf = df[df["State"] == state].sort_values("Year")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(sdf["Year"], sdf[TARGET_COL], "o-", color="#2e7d32", lw=2.5, ms=7)
    ax.fill_between(sdf["Year"], sdf[TARGET_COL], alpha=0.12, color="#2e7d32")
    ax.set(title=f"CO₂ Trend — {state}", xlabel="Year", ylabel="CO₂ (MT)")
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    return fig


def fig_forecast(hist, fore, state):
    h = hist[hist["State"] == state].sort_values("Year")
    f = fore.sort_values("Year")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(h["Year"], h[TARGET_COL], "o-", color="#2e7d32", lw=2.5, ms=7, label="Historical")
    ax.plot(f["Year"], f[TARGET_COL], "s--", color="#e65100", lw=2.5, ms=7, label="Forecast")
    ax.fill_between(f["Year"], f[TARGET_COL]*0.95, f[TARGET_COL]*1.05,
                    alpha=0.18, color="#e65100", label="±5% CI")
    ax.axvline(h["Year"].max(), color="gray", ls=":", lw=1.5)
    ax.set(title=f"Forecast — {state}", xlabel="Year", ylabel="CO₂ (MT)")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    return fig


def fig_pie(row):
    labels = ["Coal", "Gas", "Oil", "Solar", "Wind", "Hydro"]
    vals   = [row.get(c, 0) for c in [
        "Coal_Consumption","Gas_Consumption","Oil_Consumption",
        "Solar_Generation","Wind_Generation","Hydro_Generation"]]
    colors = ["#b71c1c","#f57f17","#795548","#f9a825","#1565c0","#00838f"]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.pie(vals, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=140, textprops={"fontsize": 8})
    ax.set_title("Energy Mix", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def fig_state_bar(df):
    grp = df.groupby("State")[TARGET_COL].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(grp.index, grp.values, color=PALETTE[:len(grp)])
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.set(title="Avg CO₂ by State", ylabel="CO₂ (MT)")
    ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    return fig


def fig_heatmap(df):
    corr = df[FEATURE_COLS + [TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                annot=True, fmt=".2f", cmap="YlGn", ax=ax,
                linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def fig_renewable_scatter(df):
    df2 = df.copy()
    df2["Renewable"] = (df2["Solar_Generation"] +
                        df2["Wind_Generation"] +
                        df2["Hydro_Generation"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, st in enumerate(df2["State"].unique()):
        s = df2[df2["State"] == st]
        ax.scatter(s["Renewable"], s[TARGET_COL],
                   label=st, color=PALETTE[i], s=55,
                   edgecolors="white", lw=0.5)
    x, y = df2["Renewable"].values, df2[TARGET_COL].values
    m, b = np.polyfit(x, y, 1)
    xr   = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, m*xr+b, "--", color="gray", lw=1.5, label="Trend")
    ax.set(title="Renewable vs CO₂", xlabel="Renewable (BU)", ylabel="CO₂ (MT)")
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    return fig


def fig_feature_lines(df, feat):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, st in enumerate(df["State"].unique()):
        s = df[df["State"] == st].sort_values("Year")
        ax.plot(s["Year"], s[feat], "o-", label=st, color=PALETTE[i], lw=2)
    ax.set(title=f"{feat} over Time", xlabel="Year", ylabel=feat)
    ax.legend(fontsize=7)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# MODULE 6 — UI
# ═══════════════════════════════════════════════════════════════

# Sidebar
with st.sidebar:
    st.markdown("## 🌿 Carbon Footprint System")
    st.markdown("---")
    csv_path = st.text_input("CSV path", value=CSV_DEFAULT)
    pkl_path = st.text_input("Model .pkl path", value=PKL_DEFAULT)
    st.markdown("---")
    st.caption("Keep both files in the same folder as app.py")

# Load resources
if not os.path.exists(csv_path):
    st.error(f"CSV not found: `{csv_path}`\n\nUpdate the path in the sidebar.")
    st.stop()

df             = load_data(csv_path)
model, backend = load_model(pkl_path, csv_path)
avg_em         = df[TARGET_COL].mean()
states         = sorted(df["State"].unique())

# Header
st.markdown(
    '<div class="main-header">🌍 Carbon Footprint Prediction & Reduction System</div>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Predict · Forecast · Reduce · Visualize</div>',
    unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("📊 Records",  len(df))
k2.metric("🗺️ States",   df["State"].nunique())
k3.metric("📅 Years",    f"{df['Year'].min()}–{df['Year'].max()}")
k4.metric("🏭 Avg CO₂", f"{avg_em:.1f} MT")
st.caption(f"🤖 Model backend: **{backend}**")
st.markdown("---")

# Tabs
t1, t2, t3, t4, t5 = st.tabs([
    "📂 Data", "🤖 Predict", "📈 Forecast", "♻️ Reduction", "📊 Visualize"
])

# ── TAB 1 : DATA ─────────────────────────────────────────────
with t1:
    st.subheader("Dataset Overview")
    ca, cb = st.columns([2, 1])
    with ca:
        sel_states = st.multiselect("States", states, default=states)
        yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())
        yr_range = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
    filt = df[df["State"].isin(sel_states) & df["Year"].between(*yr_range)]
    st.dataframe(filt, use_container_width=True, height=300)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(filt))
    c2.metric("Min CO₂", f"{filt[TARGET_COL].min():.1f} MT")
    c3.metric("Max CO₂", f"{filt[TARGET_COL].max():.1f} MT")
    st.markdown("### Statistical Summary")
    st.dataframe(filt[FEATURE_COLS + [TARGET_COL]].describe().round(2),
                 use_container_width=True)

# ── TAB 2 : PREDICT ──────────────────────────────────────────
with t2:
    st.subheader("CO₂ Emission Predictor")
    p_state = st.selectbox("Pre-fill from", states, key="p_state")
    p_year  = st.selectbox("Year",
                            sorted(df[df["State"] == p_state]["Year"].unique()),
                            key="p_year")
    ref = df[(df["State"] == p_state) & (df["Year"] == p_year)].iloc[0]

    st.markdown("#### Adjust feature values")
    pa, pb = st.columns(2)
    with pa:
        coal  = st.slider("Coal Consumption (BU)",  80, 300, int(ref["Coal_Consumption"]))
        gas   = st.slider("Gas Consumption (BU)",   20, 150, int(ref["Gas_Consumption"]))
        oil   = st.slider("Oil Consumption (BU)",   10, 120, int(ref["Oil_Consumption"]))
        solar = st.slider("Solar Generation (BU)",   5, 200, int(ref["Solar_Generation"]))
    with pb:
        wind  = st.slider("Wind Generation (BU)",   10, 200, int(ref["Wind_Generation"]))
        hydro = st.slider("Hydro Generation (BU)",  20, 200, int(ref["Hydro_Generation"]))
        dem   = st.slider("Energy Demand (BU)",    200, 900, int(ref["Energy_Demand"]))
        pop   = st.slider("Population (millions)",  40, 250, int(ref["Population"]))

    feats    = np.array([coal, gas, oil, solar, wind, hydro, dem, pop], dtype=float)
    pred_co2 = predict_one(model, backend, feats)
    delta_pct = (pred_co2 - avg_em) / avg_em * 100

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("🏭 Predicted CO₂", f"{pred_co2:.2f} MT")
    m2.metric("📊 Historical Avg",  f"{avg_em:.2f} MT")
    m3.metric("Δ vs Average",       f"{delta_pct:+.1f}%",
              delta=f"{delta_pct:+.1f}%", delta_color="inverse")

    if pred_co2 > avg_em * 1.2:
        level, css = "🔴 HIGH",   "alert-high"
    elif pred_co2 > avg_em * 0.9:
        level, css = "🟠 MEDIUM", "alert-med"
    else:
        level, css = "🟢 LOW",    "alert-low"
    st.markdown(f'<div class="{css}"><b>Emission Level: {level}</b></div>',
                unsafe_allow_html=True)

    st.markdown("#### Energy Mix")
    row_dict = {
        "Coal_Consumption": coal, "Gas_Consumption": gas,
        "Oil_Consumption":  oil,  "Solar_Generation": solar,
        "Wind_Generation":  wind, "Hydro_Generation": hydro,
        "Energy_Demand":    dem,
    }
    st.pyplot(fig_pie(row_dict))

# ── TAB 3 : FORECAST ─────────────────────────────────────────
with t3:
    st.subheader("Future CO₂ Forecast")
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        fc_state    = st.selectbox("State", states, key="fc_s")
        fc_yrs      = st.slider("Years ahead", 1, 10, 5)
        fc_scenario = st.radio("Scenario", [
            "Business-as-Usual",
            "Moderate Transition (−15% coal/yr)",
            "Aggressive Green (−30% coal/yr)",
        ])
    fore_df = forecast_state(df, fc_state, fc_yrs, model, backend)
    if "Moderate" in fc_scenario:
        fore_df[TARGET_COL] *= np.linspace(1.0, 0.85**fc_yrs, fc_yrs)
    elif "Aggressive" in fc_scenario:
        fore_df[TARGET_COL] *= np.linspace(1.0, 0.70**fc_yrs, fc_yrs)
    with fc2:
        st.pyplot(fig_forecast(df, fore_df, fc_state))

    st.markdown("#### Forecast Table")
    st.dataframe(
        fore_df[["Year", TARGET_COL] + FEATURE_COLS].round(2).set_index("Year"),
        use_container_width=True)

    start_em = df[df["State"] == fc_state].sort_values("Year").iloc[-1][TARGET_COL]
    end_em   = fore_df.iloc[-1][TARGET_COL]
    chg      = end_em - start_em
    st.info(
        f"**{fc_state} | {fc_scenario}**  \n"
        f"Current: **{start_em:.1f} MT** → "
        f"Projected: **{end_em:.1f} MT**  "
        f"({'📉' if chg < 0 else '📈'} {chg:+.1f} MT)")

# ── TAB 4 : REDUCTION ────────────────────────────────────────
with t4:
    st.subheader("Emission Reduction Suggestions")
    r_state = st.selectbox("State", states, key="r_s")
    r_year  = st.selectbox("Year",
                            sorted(df[df["State"] == r_state]["Year"].unique(),
                                   reverse=True),
                            key="r_y")
    r_row   = df[(df["State"] == r_state) & (df["Year"] == r_year)].iloc[0].to_dict()
    r_feats = np.array([r_row[c] for c in FEATURE_COLS], dtype=float)
    r_pred  = predict_one(model, backend, r_feats)
    r_row["CO2_Emission"] = r_pred
    sugs    = get_suggestions(r_row, avg_em)

    rd1, rd2 = st.columns([1, 2])
    with rd1:
        lvl = ("🔴 HIGH"   if r_pred > avg_em * 1.2 else
               "🟠 MEDIUM" if r_pred > avg_em * 0.9 else "🟢 LOW")
        st.metric("Predicted CO₂", f"{r_pred:.1f} MT")
        st.metric("Emission Level", lvl)
        ren = (r_row["Solar_Generation"] +
               r_row["Wind_Generation"] +
               r_row["Hydro_Generation"])
        st.metric("Renewable Share",
                  f"{ren / r_row['Energy_Demand'] * 100:.1f}%")
    with rd2:
        pri_filter = st.multiselect(
            "Priority filter", ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"])
        for s in sugs:
            if s["pri"] not in pri_filter:
                continue
            icon = "🔴" if s["pri"] == "High" else "🟠" if s["pri"] == "Medium" else "🟢"
            st.markdown(
                f'<div class="suggestion-box">'
                f'<b>{s["cat"]}</b>&nbsp;&nbsp;{icon} <b>{s["pri"]} Priority</b><br>'
                f'{s["text"]}<br>'
                f'<small>💡 <b>Impact:</b> {s["impact"]}</small></div>',
                unsafe_allow_html=True)

    hi  = sum(1 for s in sugs if s["pri"] == "High")
    pot = min(35, hi * 8)
    st.markdown(f"#### Potential reduction if all {hi} high-priority actions taken")
    pc1, pc2 = st.columns([4, 1])
    pc1.progress(pot / 100)
    pc2.write(f"**~{pot}%**")

# ── TAB 5 : VISUALIZE ────────────────────────────────────────
with t5:
    st.subheader("Visualization Dashboard")
    v1, v2 = st.columns(2)
    with v1:
        v_state = st.selectbox("State for trend", states, key="v_s")
        st.pyplot(fig_trend(df, v_state))
    with v2:
        st.pyplot(fig_state_bar(df))

    st.markdown("---")
    w1, w2 = st.columns(2)
    with w1:
        st.pyplot(fig_heatmap(df))
    with w2:
        st.pyplot(fig_renewable_scatter(df))

    st.markdown("### Feature Over Time")
    feat_sel = st.selectbox("Feature", FEATURE_COLS + [TARGET_COL])
    st.pyplot(fig_feature_lines(df, feat_sel))

# Footer
st.markdown("---")
st.caption("🌿 Carbon Footprint Prediction & Reduction System  |  "
           "Streamlit · Pandas · Scikit-learn · Matplotlib · Seaborn")