"""
Streamlit app: Battery Optimisation — Real Belgian Household
Run: cd personal_optimization && streamlit run app.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from solar_utils import optimize_day

# ── Constants ─────────────────────────────────────────────────────────────────
S_MAX, P_MAX          = 5.0, 0.75
P_KW                  = P_MAX * 4      # kW — for display / sensitivity labels
ETA_C, ETA_D          = 0.97, 0.97
S_INIT, S_MIN, DEG    = 2.5, 0.5, 0.10
PRICE_DAG, PRICE_NACHT, PRICE_INJ = 0.30, 0.22, 0.0
MARKUP      = 0.17
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "Data", "real_load", "Final")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

C = {
    "import":    "#4a90d9",
    "solar":     "#f39c12",
    "inject":    "#27ae60",
    "charge":    "#e67e22",
    "discharge": "#8e44ad",
    "sofar":     "#3498db",
    "lp_dn":     "#2ecc71",
    "lp_ep":     "#e74c3c",
    "s0":        "#95a5a6",
    "consume":   "#2c3e50",
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Battery Optimisation Belgium",
                   page_icon="🔋", layout="wide")
st.title("🔋 Battery Optimisation — Real Belgian Household")
st.caption("SOFAR ME3000SP · BYD LFP 5 kWh / 3 kW · Nov 2024 – Apr 2026")

# ── Helpers ────────────────────────────────────────────────────────────────────
def monthly_annual(series):
    s = series.dropna().copy()
    s.index = pd.to_datetime(s.index)
    avg  = s.groupby(s.index.month).mean()
    days = {1:31, 2:28.25, 3:31, 4:30, 5:31, 6:30,
            7:31,  8:31,  9:30, 10:31, 11:30, 12:31}
    return sum(avg.get(m, 0.0) * days[m] for m in range(1, 13))

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_eda_data():
    """Load full 15-min dataset and resample to hourly for EDA charts.
    Solar from optimization_input.csv (derived); SOC from overall_verrijkt.csv."""
    opt = pd.read_csv(os.path.join(DATA_DIR, "optimization_input.csv"),
                      index_col="kwartier", parse_dates=True).sort_index()
    raw = pd.read_csv(os.path.join(DATA_DIR, "overall_verrijkt.csv"),
                      parse_dates=["kwartier"], index_col="kwartier").sort_index()
    opt = opt.join(raw[["soc_begin"]], how="left")

    kwh = ["afname_kwh", "injectie_kwh", "verbruik_kwh",
           "sl_productie_kwh", "bat_laden_kwh_kw", "bat_ontladen_kwh_kw"]
    h = opt[kwh].resample("h").sum(min_count=1)
    h[["soc_begin"]] = opt[["soc_begin"]].resample("h").mean()
    h["tarief"]   = opt["tarief"].resample("h").first()
    h["hour"]     = h.index.hour
    h["price_dn"] = h["tarief"].map({"dag": PRICE_DAG, "nacht": PRICE_NACHT})
    h["cost_dn"]  = h["afname_kwh"] * h["price_dn"]
    return h

@st.cache_data
def load_data():
    """Load optimization_input.csv (used for LP validation & backtest tabs)."""
    df = pd.read_csv(os.path.join(DATA_DIR, "optimization_input.csv"),
                     index_col="kwartier", parse_dates=True)
    df["tarief_price"] = df["tarief"].map({"dag": PRICE_DAG, "nacht": PRICE_NACHT})
    df["cost_actual"]  = df["afname_kwh"] * df["tarief_price"]
    try:
        raw = pd.read_csv(os.path.join(DATA_DIR, "overall_verrijkt.csv"),
                          index_col=0, parse_dates=True)
        if "soc_begin" in raw.columns:
            df = df.join(raw[["soc_begin"]], how="left")
    except Exception:
        pass
    if "soc_begin" not in df.columns:
        df["soc_begin"] = np.nan
    # ML forecast columns
    try:
        cons_fc = pd.read_csv(os.path.join(DATA_DIR, "consumption_forecast.csv"),
                              index_col=0, parse_dates=True)
        sol_fc  = pd.read_csv(os.path.join(DATA_DIR, "solar_forecast.csv"),
                              index_col=0, parse_dates=True)
        df = df.join(cons_fc[["verbruik_fc"]], how="left")
        df = df.join(sol_fc[["sl_productie_forecast"]], how="left")
    except Exception:
        df["verbruik_fc"] = np.nan
        df["sl_productie_forecast"] = np.nan
    return df

@st.cache_data
def load_backtest_csvs():
    bt_dn = pd.read_csv(os.path.join(RESULTS_DIR, "backtest_lp_dagNacht.csv"),
                        parse_dates=["date"], index_col="date")
    bt_ep = pd.read_csv(os.path.join(RESULTS_DIR, "backtest_lp_allin.csv"),
                        parse_dates=["date"], index_col="date")
    bt_dn.index = bt_dn.index.date
    bt_ep.index = bt_ep.index.date
    return bt_dn, bt_ep

with st.spinner("Loading data…"):
    hourly          = load_eda_data()
    df              = load_data()
    df25            = df.loc["2025"].copy()
    df26            = df.loc["2026"].copy()
    bt_dn, bt_allin = load_backtest_csvs()

soc_days_26  = set(df26[df26["soc_begin"].notna()].index.normalize().date)
neg_days_26  = set(df26[df26["price_eur_kwh"] < 0].index.normalize().date)
_spread_26   = df26.groupby(df26.index.date)["price_eur_kwh"].agg(lambda x: x.max() - x.min())
spread_days_26   = set(_spread_26[_spread_26 > 0.15].index)
interesting_26   = sorted(soc_days_26 | neg_days_26 | spread_days_26)

# ── Available dates for ML Forecast tab (2026 only, SOC recorded at 13:00) ────
import datetime as _dt
_df26_at13      = df.loc[(df.index.year == 2026) & (df.index.hour == 13) & (df.index.minute == 0)]
_ml_avail_dates = sorted(_df26_at13[_df26_at13["soc_begin"].notna()].index.date.tolist())

# Fixed y-axis range for EPEX price panel (global, across entire 2026 dataset)
_ep_min = float(df26["price_eur_kwh"].min())
_ep_max = float(df26["price_eur_kwh"].max())
EPEX_PRICE_YLIM = (
    round(min(_ep_min + MARKUP, -0.02) - 0.04, 2),
    round(_ep_max + MARKUP + 0.06, 2),
)

# ── SOFAR daily costs for scenario comparison ──────────────────────────────────
afname_no_ev   = (df["afname_kwh"] - df["ev_energie_kwh"].fillna(0)).clip(lower=0)
sofar_wear_day = df["bat_laden_kwh_kw"].clip(lower=0, upper=0.75).resample("D").sum() * DEG
sofar_wear_day.index = sofar_wear_day.index.date
sofar_dn_d = (afname_no_ev * df["tarief_price"]).resample("D").sum() \
           - (df["injectie_kwh"] * PRICE_INJ).resample("D").sum()
sofar_dn_d.index = sofar_dn_d.index.date
sofar_ep_d = (afname_no_ev * (df["price_eur_kwh"] + MARKUP)).resample("D").sum() \
           - (df["injectie_kwh"] * PRICE_INJ).resample("D").sum()
sofar_ep_d.index = sofar_ep_d.index.date

common     = bt_dn.index.intersection(bt_allin.index).intersection(sofar_dn_d.index)
ann_s0_dn  = monthly_annual(bt_dn["cost_baseline"].reindex(common))
ann_sf_dn  = monthly_annual(sofar_dn_d.reindex(common))
ann_sw_dn  = monthly_annual(sofar_wear_day.reindex(common))
ann_lp_dn  = monthly_annual(bt_dn["cost"].reindex(common))
ann_s0_ep  = monthly_annual(bt_allin["cost_baseline"].reindex(common))
ann_sf_ep  = monthly_annual(sofar_ep_d.reindex(common))
ann_sw_ep  = ann_sw_dn
ann_lp_ep  = monthly_annual(bt_allin["cost"].reindex(common))
yr_s1w     = ann_sf_dn + ann_sw_dn
yr_s4w     = ann_sf_ep + ann_sw_ep

# ── Hourly EPEX forecast profile (mean price by hour, computed from full dataset) ─
_epex_by_hour = df["price_eur_kwh"].groupby(df.index.hour).mean()

# ── Run LP (validation tab — dag/nacht, or EPEX with perfect prices) ──────────
def run_lp(data: pd.DataFrame, use_epex: bool, markup: float) -> pd.DataFrame:
    rows, s = [], S_INIT
    for date in sorted(set(data.index.date)):
        day = data.loc[str(date)]
        p   = (day["price_eur_kwh"] + markup).values if use_epex else day["tarief_price"].values
        l   = day["verbruik_kwh"].values
        sol = day["sl_productie_kwh"].values
        T   = len(p)
        if T not in [92, 96, 100]:
            continue
        res = optimize_day(p, l, S_MAX, P_MAX, ETA_C, ETA_D, s,
                           cyclic=False, binary=True, deg_cost=DEG,
                           S_min=S_MIN, solar=sol, price_inj=PRICE_INJ)
        for t in range(T):
            rows.append(dict(ts=day.index[t], price=p[t], solar=sol[t],
                             c=res["c"][t], d=res["d"][t], s=res["s"][t],
                             g_in=res["g_in"][t], g_out=res["g_out"][t]))
        s = res["s_final"]
    return pd.DataFrame(rows).set_index("ts") if rows else pd.DataFrame()


@st.cache_data(show_spinner="Running LP optimisation (~30 s)…")
def _lp_custom(s_max: float, p_kw: float, deg: float, markup: float) -> pd.DataFrame:
    """Run full LP backtest with custom battery parameters (binary=False for speed)."""
    data   = load_data()
    p_max  = p_kw / 4.0
    s_min  = 0.10 * s_max
    rows, s = [], s_max * 0.50
    for date in sorted(set(data.index.date)):
        day = data.loc[str(date)]
        p   = (day["price_eur_kwh"] + markup).values
        l   = day["verbruik_kwh"].values
        sol = day["sl_productie_kwh"].values
        T   = len(p)
        if T not in [92, 96, 100]:
            continue
        p_dn      = day["tarief_price"].values
        net_load  = np.maximum(l - sol, 0)
        # Baseline A: solar-only, dag/nacht tariff, NO battery
        cost_b_dn = float(np.dot(p_dn, net_load))
        # Baseline B: solar-only, EPEX+markup tariff, NO battery (intermediate)
        cost_b_ep = float(np.dot(p,    net_load))
        res = optimize_day(p, l, s_max, p_max, ETA_C, ETA_D, s,
                           cyclic=False, binary=False, deg_cost=deg,
                           S_min=s_min, solar=sol, price_inj=PRICE_INJ)
        rows.append({"date":            date,
                     "cost":            res["cost"],
                     "cost_baseline":   cost_b_dn,
                     "cost_baseline_ep": cost_b_ep,
                     "cost_deg":        res["cost_degradation"]})
        s = res["s_final"]
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
tab_eda, tab_val, tab_bt, tab_fc, tab_calc, tab_ml = st.tabs([
    "📊 EDA — Real data",
    "🔍 Validation 2026 — Real vs LP",
    "📈 Backtest — Scenario comparison",
    "⚡ LP Optimisation",
    "🔋 Battery Calculator",
    "🤖 ML Forecast LP",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA  (full period: Nov 2024 – Apr 2026)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    period_start = hourly.index.min().date()
    period_end   = hourly.index.max().date()
    st.subheader(f"EDA — Real household data  ({period_start} → {period_end})")
    st.caption(f"SOFAR ME3000SP · BYD LFP 5 kWh — {len(hourly)} hourly observations")

    # ── Top metrics (all data) ─────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Grid import",        f"{hourly['afname_kwh'].sum():.0f} kWh")
    m2.metric("Solar delivered",    f"{hourly['sl_productie_kwh'].sum():.0f} kWh")
    m3.metric("Grid injection",     f"{hourly['injectie_kwh'].sum():.0f} kWh")
    m4.metric("Battery charged",    f"{hourly['bat_laden_kwh_kw'].sum():.0f} kWh")
    m5.metric("Total cost (dag/n)", f"{hourly['cost_dn'].sum():.0f} EUR")

    st.divider()

    # Pre-compute shared data
    monthly_h = hourly[["afname_kwh", "injectie_kwh", "bat_laden_kwh_kw",
                         "bat_ontladen_kwh_kw", "verbruik_kwh",
                         "sl_productie_kwh", "cost_dn"]].resample("ME").sum(min_count=1)
    monthly_h.index = monthly_h.index.strftime("%b %Y")
    x_m = range(len(monthly_h))
    by_hour = hourly.groupby("hour")[["afname_kwh", "injectie_kwh", "sl_productie_kwh",
                                      "bat_laden_kwh_kw", "bat_ontladen_kwh_kw",
                                      "soc_begin"]].mean()

    eda_t1, eda_t2, eda_t3, eda_t4 = st.tabs([
        "📅 Monthly overview", "⏰ Day profile",
        "🔋 Battery analysis", "🕐 Battery by tariff",
    ])

    # ── §3b + §11: Monthly overview ────────────────────────────────────────────
    with eda_t1:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Monthly energy flows**")
            fig, ax = plt.subplots(figsize=(7, 4))
            w_m = 0.22
            ax.bar([i - w_m for i in x_m], monthly_h["afname_kwh"],       w_m,
                   color=C["import"],  label="Grid import")
            ax.bar([i       for i in x_m], monthly_h["injectie_kwh"],     w_m,
                   color=C["inject"],  label="Grid export")
            ax.bar([i + w_m for i in x_m], monthly_h["verbruik_kwh"],     w_m,
                   color=C["consume"], label="Consumption", alpha=0.55)
            ax.set_ylabel("kWh/month")
            ax.set_xticks(list(x_m))
            ax.set_xticklabels(list(monthly_h.index), rotation=45, ha="right", fontsize=7)
            ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_r:
            st.markdown("**Monthly electricity cost (dag/nacht)**")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(x_m, monthly_h["cost_dn"], color="#c0392b", alpha=0.85)
            ax.set_ylabel("EUR / month")
            ax.set_xticks(list(x_m))
            ax.set_xticklabels(list(monthly_h.index), rotation=45, ha="right", fontsize=7)
            for xi, v in zip(x_m, monthly_h["cost_dn"]):
                ax.text(xi, v + 0.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.markdown("**Monthly battery throughput**")
            bat_m = hourly[["bat_laden_kwh_kw", "bat_ontladen_kwh_kw"]].resample("ME").sum(min_count=1)
            fig, ax = plt.subplots(figsize=(7, 4))
            x6 = range(len(bat_m))
            ax.bar([i - 0.2 for i in x6], bat_m["bat_laden_kwh_kw"],    0.4,
                   color=C["charge"],    label="Charge")
            ax.bar([i + 0.2 for i in x6], bat_m["bat_ontladen_kwh_kw"], 0.4,
                   color=C["discharge"], label="Discharge")
            ax.set_xticks(list(x6))
            ax.set_xticklabels(bat_m.index.strftime("%b %Y"), rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("kWh/month"); ax.legend(fontsize=8)
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_r2:
            st.markdown("**Monthly solar & injection**")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(x_m, monthly_h["sl_productie_kwh"], color=C["solar"],  alpha=0.85, label="Solar delivered")
            ax.bar(x_m, monthly_h["injectie_kwh"],     color=C["inject"], alpha=0.85, label="Solar injection")
            ax.set_ylabel("kWh/month")
            ax.set_xticks(list(x_m))
            ax.set_xticklabels(list(monthly_h.index), rotation=45, ha="right", fontsize=7)
            ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.info(
            "**Key observations.** Consumption is relatively stable (~300–450 kWh/month). "
            "Grid import peaks in winter (Dec–Feb) when solar output is near zero. "
            "Spring (Mar–May) shows the highest solar production and battery throughput. "
            "Electricity cost is lower in summer thanks to solar self-consumption."
        )

    # ── §4: Average day profile ────────────────────────────────────────────────
    with eda_t2:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Average hourly energy flows**")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(by_hour.index, by_hour["afname_kwh"],       color=C["import"],
                    marker="o", ms=3, label="Grid import")
            ax.plot(by_hour.index, by_hour["injectie_kwh"],     color=C["inject"],
                    marker="o", ms=3, label="Grid export")
            ax.plot(by_hour.index, by_hour["sl_productie_kwh"], color=C["solar"],
                    marker="o", ms=3, label="Solar")
            ax.set_ylabel("Mean kWh/h"); ax.set_xlabel("Hour")
            ax.set_xticks(range(0, 24, 2)); ax.legend(fontsize=8)
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_r:
            st.markdown("**Average hourly battery charge / discharge**")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(by_hour.index - 0.2, by_hour["bat_laden_kwh_kw"],    0.4,
                   color=C["charge"],    label="Charge")
            ax.bar(by_hour.index + 0.2, by_hour["bat_ontladen_kwh_kw"], 0.4,
                   color=C["discharge"], label="Discharge")
            ax.set_ylabel("Mean kWh/h"); ax.set_xlabel("Hour")
            ax.set_xticks(range(0, 24, 2)); ax.legend(fontsize=8)
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        soc_by_h = by_hour["soc_begin"].dropna()
        if len(soc_by_h) > 0:
            col_soc, _ = st.columns([2, 1])
            with col_soc:
                st.markdown(f"**Average SOC by hour (Jan–Apr 2026, {len(soc_days_26)} logged days)**")
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(soc_by_h.index, soc_by_h.values, color="#c0392b", marker="o", ms=4)
                ax.fill_between(soc_by_h.index, soc_by_h.values, alpha=0.18, color="#c0392b")
                ax.set_ylabel("Mean SOC (%)"); ax.set_xlabel("Hour")
                ax.set_ylim(0, 105); ax.set_xticks(range(0, 24, 2))
                ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        st.info(
            "**Key observations.** Solar generation peaks at 11:00–14:00. "
            "Battery charges at night (22:00–06:00, cheap nacht tariff) and around noon from solar surplus. "
            "Discharge happens mainly during the day (07:00–22:00) at the higher dag tariff. "
            "SOC is highest in the morning (~60–70%) after night charging, then drops through the day."
        )

    # ── §6: Battery analysis ───────────────────────────────────────────────────
    with eda_t3:
        bat_monthly = hourly[["bat_laden_kwh_kw", "bat_ontladen_kwh_kw"]].resample("ME").sum(min_count=1)
        total_charge = hourly["bat_laden_kwh_kw"].sum()
        n_days_total = (hourly.index.max() - hourly.index.min()).days
        efc = total_charge / S_MAX
        efc_yr = efc / n_days_total * 365
        years_life = 6000 / efc_yr if efc_yr > 0 else 0

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Monthly battery throughput**")
            fig, ax = plt.subplots(figsize=(7, 4))
            x6 = range(len(bat_monthly))
            ax.bar([i - 0.2 for i in x6], bat_monthly["bat_laden_kwh_kw"],    0.4,
                   color=C["charge"],    label="Charge")
            ax.bar([i + 0.2 for i in x6], bat_monthly["bat_ontladen_kwh_kw"], 0.4,
                   color=C["discharge"], label="Discharge")
            ax.set_xticks(list(x6))
            ax.set_xticklabels(bat_monthly.index.strftime("%b %Y"),
                               rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("kWh/month"); ax.legend(fontsize=8)
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_r:
            st.markdown("**SOC distribution (Jan–Apr 2026)**")
            soc_valid = hourly["soc_begin"].dropna()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(soc_valid, bins=25, color="#c0392b", edgecolor="white", linewidth=0.3)
            ax.axvline(soc_valid.mean(), color="navy", lw=1.5,
                       label=f"Mean = {soc_valid.mean():.1f}%")
            ax.set_xlabel("Battery SOC (%)"); ax.set_ylabel("Hours")
            ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.info(
            f"**Battery wear**  |  "
            f"Total charged: {total_charge:.0f} kWh over {n_days_total} days "
            f"({total_charge / n_days_total:.2f} kWh/day avg)  |  "
            f"EFC = {total_charge:.0f} kWh ÷ {S_MAX:.0f} kWh (usable capacity) = **{efc:.0f} EFC total**  |  "
            f"→ **{efc_yr:.0f} EFC/year**  |  "
            f"Warranty 6 000 EFC → estimated cycle life **{years_life:.0f} years** "
            f"(calendar warranty of 10 years will apply first)."
        )
        st.info(
            f"**Key observations.** Throughput peaks in spring (Mar–Apr) — solar charges the battery by day, "
            f"the nacht tariff charges it at night. "
            f"SOC distribution: battery spends most time at 20–60%, rarely fully charged. "
            f"At {efc_yr:.0f} EFC/year the battery is used gently — "
            f"well within the 6 000 EFC warranty limit over a 10-year horizon."
        )

    # ── §7b: Battery behavior by tariff ───────────────────────────────────────
    with eda_t4:
        df25_qh = df.loc["2025"].copy()
        df25_qh["hour"] = df25_qh.index.hour
        by_h_25 = df25_qh.groupby("hour")[
            ["bat_laden_kwh_kw", "bat_ontladen_kwh_kw", "tarief"]
        ].agg(
            laden   =("bat_laden_kwh_kw",    "mean"),
            ontladen=("bat_ontladen_kwh_kw", "mean"),
            tarief  =("tarief", lambda x: x.value_counts().idxmax()),
        )
        colors_l = ["#1a6bb5" if t == "nacht" else "#e07b00" for t in by_h_25["tarief"]]
        colors_d = ["#7b3fa0" if t == "dag"   else "#a0c0e0" for t in by_h_25["tarief"]]

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Battery charging by hour — colored by tariff**")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(by_h_25.index, by_h_25["laden"] * 1000, color=colors_l,
                   edgecolor="white", linewidth=0.4)
            ax.axvline(6,  color="gray", lw=1, ls="--", alpha=0.6)
            ax.axvline(22, color="gray", lw=1, ls="--", alpha=0.6)
            ax.set_ylabel("Avg charge (Wh / 15 min)"); ax.set_xlabel("Hour")
            ax.set_xticks(range(0, 24, 2))
            ax.legend(handles=[
                mpatches.Patch(color="#1a6bb5", label="nacht (22–06)"),
                mpatches.Patch(color="#e07b00", label="dag (06–22)"),
            ], fontsize=8, title="Tariff slot")
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_r:
            st.markdown("**Battery discharging by hour — colored by tariff**")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(by_h_25.index, by_h_25["ontladen"] * 1000, color=colors_d,
                   edgecolor="white", linewidth=0.4)
            ax.axvline(6,  color="gray", lw=1, ls="--", alpha=0.6)
            ax.axvline(22, color="gray", lw=1, ls="--", alpha=0.6)
            ax.set_ylabel("Avg discharge (Wh / 15 min)"); ax.set_xlabel("Hour")
            ax.set_xticks(range(0, 24, 2))
            ax.legend(handles=[
                mpatches.Patch(color="#7b3fa0", label="dag (06–22)"),
                mpatches.Patch(color="#a0c0e0", label="nacht (22–06)"),
            ], fontsize=8, title="Tariff slot")
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.info(
            "**Key observations.** SOFAR charges **at night** (22:00–06:00, blue bars) from cheap grid power "
            "and around midday from solar surplus (orange bars). "
            "Discharge happens almost exclusively **during the day** (06:00–22:00, purple) at the higher dag tariff. "
            "Core logic: buy cheap at night → use (offset) expensive daytime grid. "
            "The small discharge visible at 22:00–24:00 is residual household consumption."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VALIDATION 2026  (3 scenarios side-by-side)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_val:
    st.subheader("Real SOFAR vs LP optimal — three scenarios side by side")
    st.caption(
        f"Only **interesting days** shown: ⭐ SOC logged · ⚡ negative EPEX price · 📈 EPEX spread > 0.15 EUR/kWh "
        f"({len(interesting_26)} days total).  "
        f"Belgian weekend/holiday = nacht all day ({PRICE_NACHT} EUR/kWh) — flat price line is correct.  \n"
        "**★ LP EPEX = theoretical best case**: uses actual historical prices as a stand-in for a perfect "
        "price forecast — shows the upper bound of what dynamic pricing could achieve.  "
        "🟢 price published day-ahead (realistic)  ·  🟠 multi-day forecast would be needed here."
    )

    ctrl, _ = st.columns([1, 3])
    with ctrl:
        def _fmt_date(d):
            tags = []
            if d in soc_days_26:  tags.append("⭐")
            if d in neg_days_26:  tags.append("⚡")
            if d in spread_days_26: tags.append("📈")
            return f"{''.join(tags)}  {d}" if tags else str(d)

        selected = st.selectbox(
            "Start date  (⭐ SOC  ⚡ neg price  📈 spread)",
            options=interesting_26,
            format_func=_fmt_date,
        )
        n_days = st.radio("Days to show", [1, 2, 3, 4], horizontal=True)
        markup = st.slider("EPEX markup (EUR/kWh)", 0.10, 0.25, MARKUP, 0.01)

    idx       = interesting_26.index(selected)
    dates_sel = interesting_26[idx : min(idx + n_days, len(interesting_26))]
    sl        = df26[np.isin(df26.index.date, dates_sel)].copy()

    if sl.empty:
        st.warning("No data for selected period.")
    else:
        with st.spinner("Running LP (both strategies)…"):
            lp_dn = run_lp(sl, use_epex=False, markup=0.0)
            lp_ep = run_lp(sl, use_epex=True,  markup=markup)  # actual prices = perfect-forecast proxy

        # Real SOFAR approx SOC via cumulative integration
        bat_c_real = sl["bat_laden_kwh_kw"].clip(lower=0, upper=P_MAX)
        bat_d_real = sl["bat_ontladen_kwh_kw"].clip(lower=0)
        s_real     = np.clip(S_INIT + np.cumsum((bat_c_real - bat_d_real).values), 0.0, S_MAX)

        # Daily costs
        afname_no_ev_sl = (sl["afname_kwh"] - sl["ev_energie_kwh"].fillna(0)).clip(lower=0)
        cost_real_d = (
            (afname_no_ev_sl * sl["tarief_price"]).resample("D").sum()
            - (sl["injectie_kwh"] * PRICE_INJ).resample("D").sum()
            + (bat_c_real * DEG).resample("D").sum()
        )

        def _lp_day_cost(lp_df):
            if lp_df.empty:
                return pd.Series(dtype=float)
            return (
                (lp_df["price"] * lp_df["g_in"]).resample("D").sum()
                - (PRICE_INJ * lp_df["g_out"]).resample("D").sum()
                + (DEG * lp_df["c"]).resample("D").sum()
            )

        cost_dn_d = _lp_day_cost(lp_dn)
        cost_ep_d = _lp_day_cost(lp_ep)

        # ── Cost metrics at the top ────────────────────────────────────────────
        total_real = float(cost_real_d.sum())
        total_dn   = float(cost_dn_d.sum()) if not cost_dn_d.empty else 0.0
        total_ep   = float(cost_ep_d.sum()) if not cost_ep_d.empty else 0.0
        period_label = (f"{dates_sel[0]} — {dates_sel[-1]}"
                        if len(dates_sel) > 1 else str(dates_sel[0]))
        st.markdown(f"**Period cost ({period_label}, incl. battery wear)**")
        _pct_dn = (total_real - total_dn) / total_real * 100 if total_real else 0
        _pct_ep = (total_real - total_ep) / total_real * 100 if total_real else 0
        _pct_ep_vs_dn = (total_dn - total_ep) / total_dn * 100 if total_dn else 0
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Real SOFAR",       f"{total_real:.2f} EUR")
        cc2.metric("LP dag/nacht",     f"{total_dn:.2f} EUR",
                   delta=f"{total_real - total_dn:+.2f} EUR  ({_pct_dn:+.1f}% vs Real)")
        cc3.metric("LP EPEX + markup", f"{total_ep:.2f} EUR",
                   delta=f"{total_real - total_ep:+.2f} EUR  ({_pct_ep:+.1f}% vs Real · {_pct_ep_vs_dn:+.1f}% vs dag/nacht)")

        st.divider()

        # ── Shared plot data ───────────────────────────────────────────────────
        ts       = sl.index
        solar_kw = sl["sl_productie_kwh"].values * 4
        load_kw  = sl["verbruik_kwh"].values * 4
        p_dn_arr = sl["tarief_price"].values
        p_ep_arr = (sl["price_eur_kwh"] + markup).values
        sep_times = [pd.Timestamp(d) for d in dates_sel[1:]]

        real_net_kw = (bat_c_real - bat_d_real).values * 4
        dn_net = (lp_dn["c"] - lp_dn["d"]).values * 4 if not lp_dn.empty else np.zeros(len(ts))
        dn_soc = lp_dn["s"].values if not lp_dn.empty else np.full(len(ts), S_INIT)
        dn_idx = lp_dn.index if not lp_dn.empty else ts
        ep_net = (lp_ep["c"] - lp_ep["d"]).values * 4 if not lp_ep.empty else np.zeros(len(ts))
        ep_soc = lp_ep["s"].values if not lp_ep.empty else np.full(len(ts), S_INIT)
        ep_idx = lp_ep.index if not lp_ep.empty else ts

        # ── Shared y-limits (same scale across all 3 columns) ─────────────────
        # Single fixed price range for both panels: starts at 0 (or lower if EPEX goes negative)
        _shared_price_lo = min(0.0, EPEX_PRICE_YLIM[0])
        _shared_price_hi = max(PRICE_DAG + 0.04, EPEX_PRICE_YLIM[1])
        price_dn_ylim = (_shared_price_lo, _shared_price_hi)
        price_ep_ylim = (_shared_price_lo, _shared_price_hi)
        _solar_top    = max(float(solar_kw.max()), float(load_kw.max())) * 1.15
        solar_ylim    = (0.0, max(_solar_top, 0.2))
        _all_net      = np.concatenate([real_net_kw, dn_net, ep_net])
        _bat_abs      = max(float(np.abs(_all_net).max()) * 1.25, 0.4)
        bat_ylim      = (-_bat_abs, _bat_abs)
        soc_ylim      = (-0.3, S_MAX + 0.8)

        # ── Styling helpers ────────────────────────────────────────────────────
        _CHARGE_COL    = "#2ecc71"   # green for all charge bars
        _DISCHARGE_COL = "#e74c3c"   # red for all discharge bars
        _GRID_KW       = dict(alpha=0.25, lw=0.7, color="#888")
        _SPINE_ALPHA   = 0.25

        def _style_ax(ax):
            """Remove top/right spines, soften remaining ones, clean grid."""
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_alpha(_SPINE_ALPHA)
            ax.spines["bottom"].set_alpha(_SPINE_ALPHA)
            ax.yaxis.grid(True, **_GRID_KW)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8, length=3, color="#888")

        def _panel_fmt(ax, show_xticks=False):
            _style_ax(ax)
            for sep in sep_times:
                ax.axvline(sep, color="#aaa", lw=0.8, ls="--", alpha=0.6)
            if show_xticks:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
                ax.xaxis.grid(True, **_GRID_KW)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

        def _make_col_fig(title, title_color, prices, net_kw, soc_vals, idx_,
                          is_epex=False):
            fig, axes = plt.subplots(4, 1, figsize=(5.5, 14), sharex=True)
            fig.patch.set_facecolor("#fafafa")
            for ax in axes:
                ax.set_facecolor("#fafafa")

            # Title bar
            fig.text(0.5, 1.005, title, ha="center", va="bottom",
                     fontsize=12, fontweight="bold", color=title_color,
                     transform=fig.transFigure)

            time_idx = idx_ if idx_ is not None else ts

            # ── Panel 0: Prices ───────────────────────────────────────────────
            ax = axes[0]
            if is_epex:
                # ── Day-ahead publication shading ─────────────────────────────
                # EPEX publishes next-day prices at ~13:00 each day.
                # t_pub = 13:00 of first selected day = when day2 prices appear.
                t_pub       = pd.Timestamp(dates_sel[0]) + pd.Timedelta(hours=13)
                t_end_data  = ts[-1]  + pd.Timedelta(minutes=15)

                # Zone A: 00:00→13:00 day1  — current day, prices already known
                ax.axvspan(ts[0], min(t_pub, t_end_data),
                           color="#888888", alpha=0.07, zorder=0)

                # Zone B: 13:00 day1 → 13:00 day1+24 h  — day-ahead window (green)
                t_win_end = t_pub + pd.Timedelta(hours=24)
                ax.axvspan(t_pub, min(t_win_end, t_end_data),
                           color="#27ae60", alpha=0.14, zorder=0)

                # Zone C: beyond +24 h from 13:00  — forecast territory (orange)
                if t_win_end < t_end_data:
                    ax.axvspan(t_win_end, t_end_data,
                               color="#e67e22", alpha=0.22, zorder=0)

                # Vertical dashed lines at 13:00 of each selected day
                for d in dates_sel:
                    t_line = pd.Timestamp(d) + pd.Timedelta(hours=13)
                    if ts[0] < t_line < ts[-1]:
                        ax.axvline(t_line, color="#27ae60", lw=0.9,
                                   ls=":", alpha=0.8, zorder=4)

                ax.plot(ts, prices, color="#c0392b", lw=1.4, zorder=3)
                ax.axhline(0, color="#888", lw=0.7, ls=":")
                ax.fill_between(ts, prices, 0,
                                where=(prices < 0), color="#e74c3c",
                                alpha=0.20, zorder=2)
                ax.set_ylim(*price_ep_ylim)
                handles = [
                    mpatches.Patch(color="#888888", alpha=0.40,
                                   label="current day (known)"),
                    mpatches.Patch(color="#27ae60", alpha=0.55,
                                   label="+24 h  day-ahead pub 13:00"),
                    mpatches.Patch(color="#e67e22", alpha=0.60,
                                   label=">+24 h  forecast needed"),
                ]
                ax.legend(handles=handles, fontsize=6, loc="upper right",
                          framealpha=0.6, edgecolor="none")
                ax.set_title("EPEX + markup  ★ theoretical best case", fontsize=9, pad=4)
            else:
                ax.step(ts, prices, color="#2471a3", lw=1.5, where="post")
                ax.axhline(PRICE_DAG,   color="#2471a3", lw=0.8, ls="--",
                           alpha=0.5, label=f"dag  {PRICE_DAG:.2f}")
                ax.axhline(PRICE_NACHT, color="#85c1e9", lw=0.8, ls="--",
                           alpha=0.5, label=f"nacht {PRICE_NACHT:.2f}")
                ax.set_ylim(*price_dn_ylim)
                ax.legend(fontsize=7, loc="upper right",
                          framealpha=0.5, edgecolor="none")
                ax.set_title("Electricity price  ·  always known", fontsize=9, pad=4)
            ax.set_ylabel("EUR/kWh", fontsize=8)
            _panel_fmt(ax)

            # ── Panel 1: Solar + load ─────────────────────────────────────────
            ax = axes[1]
            ax.fill_between(ts, solar_kw, color="#f39c12", alpha=0.45, label="Solar")
            ax.plot(ts, load_kw, color="#2c3e50", lw=1.1, label="Load")
            ax.set_ylim(*solar_ylim)
            ax.set_ylabel("kW", fontsize=8)
            ax.set_title("Solar production & load", fontsize=9, pad=4)
            ax.legend(fontsize=7, loc="upper right",
                      framealpha=0.5, edgecolor="none")
            _panel_fmt(ax)

            # ── Panel 2: Battery net (charge green / discharge red) ───────────
            ax = axes[2]
            ax.fill_between(time_idx, np.maximum(net_kw, 0),
                            color=_CHARGE_COL,    alpha=0.75, label="Charge")
            ax.fill_between(time_idx, np.minimum(net_kw, 0),
                            color=_DISCHARGE_COL, alpha=0.75, label="Discharge")
            ax.axhline(0, color="#555", lw=0.7)
            ax.set_ylim(*bat_ylim)
            ax.set_ylabel("kW", fontsize=8)
            ax.set_title("Battery schedule  (+ charge / − discharge)", fontsize=9, pad=4)
            ax.legend(fontsize=7, loc="upper right",
                      framealpha=0.5, edgecolor="none")
            _panel_fmt(ax)

            # ── Panel 3: SOC (kWh) ────────────────────────────────────────────
            ax = axes[3]
            ax.plot(time_idx, soc_vals, color=title_color, lw=1.8)
            ax.fill_between(time_idx, soc_vals, alpha=0.12, color=title_color)
            ax.axhline(S_MIN, color="#e67e22", lw=0.9, ls="--",
                       label=f"S_min  {S_MIN} kWh")
            ax.axhline(S_MAX, color="#27ae60", lw=0.9, ls="--",
                       label=f"S_max {S_MAX} kWh")
            ax.set_ylim(*soc_ylim)
            ax.set_ylabel("kWh", fontsize=8)
            ax.set_title("State of charge", fontsize=9, pad=4)
            ax.legend(fontsize=7, loc="lower right",
                      framealpha=0.5, edgecolor="none")
            _panel_fmt(ax, show_xticks=True)

            fig.subplots_adjust(hspace=0.38, top=0.97, bottom=0.07,
                                left=0.14, right=0.97)
            return fig

        # ── Three columns ──────────────────────────────────────────────────────
        col_r, col_dn, col_ep = st.columns(3)
        with col_r:
            fig = _make_col_fig("Real SOFAR", "#555555",
                                p_dn_arr, real_net_kw, s_real, None)
            st.pyplot(fig); plt.close()
        with col_dn:
            fig = _make_col_fig("LP  dag/nacht", "#2471a3",
                                p_dn_arr, dn_net, dn_soc, dn_idx)
            st.pyplot(fig); plt.close()
        with col_ep:
            fig = _make_col_fig(f"LP  EPEX  +{markup:.2f}  ★ theoretical best case", "#c0392b",
                                p_ep_arr, ep_net, ep_soc, ep_idx, is_epex=True)
            st.pyplot(fig); plt.close()

        # ── Daily cost bar chart — compact, side legend ────────────────────────
        all_days = cost_real_d.index
        n_days_plot = len(all_days)
        x_pos = np.arange(n_days_plot)
        w = 0.18
        fig, ax = plt.subplots(figsize=(max(3.2, n_days_plot * 0.85), 2.4))
        fig.patch.set_facecolor("#fafafa")
        ax.set_facecolor("#fafafa")
        ax.bar(x_pos - w, cost_real_d.values, w, color="#7f8c8d",
               label="Real SOFAR", zorder=3)
        if not cost_dn_d.empty:
            ax.bar(x_pos, cost_dn_d.reindex(all_days).fillna(0).values, w,
                   color="#2471a3", label="LP dag/nacht", zorder=3)
        if not cost_ep_d.empty:
            ax.bar(x_pos + w, cost_ep_d.reindex(all_days).fillna(0).values, w,
                   color="#c0392b", label="LP EPEX ★", zorder=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [pd.Timestamp(d).strftime("%d %b") for d in all_days],
            rotation=0, ha="center", fontsize=7,
        )
        ax.axhline(0, color="#555", lw=0.6)
        ax.set_ylabel("EUR / day", fontsize=7)
        ax.set_title("Daily electricity cost  (incl. battery wear)",
                     fontsize=8, pad=5, color="#333")
        ax.legend(fontsize=6.5, ncol=1, framealpha=0.0, edgecolor="none",
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        _style_ax(ax)
        ax.tick_params(labelsize=7, length=2)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, **_GRID_KW, zorder=0)
        plt.tight_layout(pad=0.5, rect=[0, 0, 0.83, 1])
        _, bar_col, _ = st.columns([1, 2, 1])
        with bar_col:
            st.pyplot(fig, use_container_width=False); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST (§4 + §5 from 03_optimization_solar.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.subheader("Backtest — Scenario comparison & monthly results")
    st.info(
        f"Pre-computed LP backtest: **{len(common)} days** "
        f"({common[0]} → {common[-1]}).  "
        f"EPEX all-in markup = **{MARKUP} EUR/kWh**.  "
        f"All costs include battery wear (DEG = {DEG} EUR/kWh charged)."
    )

    # ── §4 Scenario comparison ─────────────────────────────────────────────────
    st.markdown("### Scenario comparison (annualised)")

    SC = {"s0":"#95a5a6", "s1w":C["sofar"], "s2":"#2980b9",
          "s3b":"#95a5a6", "s4w":"#c0392b", "s3":"#922b21"}

    def _add_arrow(ax, x1, x2, yref, diff, fs=9):
        color = "#c0392b" if diff > 0 else "#1a9c4e"
        sign  = "+" if diff > 0 else ""
        ax.annotate("", xy=(x2, yref), xytext=(x1, yref),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.5))
        ax.text((x1 + x2) / 2, yref * 1.010, f"{sign}{diff:.0f} EUR/yr",
                ha="center", fontsize=fs, fontweight="bold", color=color)

    def _bar_panel(ax, labels, values, colors, title, note):
        rng = (max(values) - min(values)) or max(values) * 0.05
        ax.bar(range(len(values)), values, color=colors, width=0.40,
               edgecolor="white", linewidth=1.0)
        for i, v in enumerate(values):
            ax.text(i, v + rng * 0.03, f"{v:.0f}", ha="center",
                    fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("EUR / year", fontsize=9)
        ax.set_ylim(min(values) * 0.88, max(values) * 1.24)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.text(0.5, -0.24, note, transform=ax.transAxes,
                ha="center", fontsize=8, color="#444", style="italic")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Scenario comparison  |  {len(common)} days ({common[0]} → {common[-1]})  |  "
        f"DEG = {DEG} EUR/kWh  |  EPEX markup = {MARKUP} EUR/kWh\n"
        "All costs include battery wear — fair comparison.",
        fontsize=10, fontweight="bold"
    )

    ax = axes[0, 0]
    vals = [ann_s0_dn, yr_s1w, ann_lp_dn]
    lbls = ["S0  Solar only\n(no battery)",
            "S1  Real SOFAR + wear\n★ TODAY",
            "S2  LP dag/nacht + wear"]
    _bar_panel(ax, lbls, vals, [SC["s0"], SC["s1w"], SC["s2"]],
        "Q1: Does the battery save money?  (dag/nacht tariff)",
        f"Real SOFAR saves {ann_s0_dn - yr_s1w:.0f} EUR/yr vs solar-only.  "
        f"LP would save an extra {yr_s1w - ann_lp_dn:.0f} EUR/yr vs real SOFAR.")
    _add_arrow(ax, 0, 1, max(vals) * 1.10, yr_s1w  - ann_s0_dn)
    _add_arrow(ax, 1, 2, max(vals) * 1.17, ann_lp_dn - yr_s1w)

    ax = axes[0, 1]
    vals = [yr_s1w, yr_s4w]
    lbls = ["S1  Real SOFAR + wear\ndag/nacht  ★ TODAY",
            "S4  Real SOFAR + wear\nEPEX + markup\n(same schedule)"]
    _bar_panel(ax, lbls, vals, [SC["s1w"], SC["s4w"]],
        "Q2: Switch to EPEX without a new controller?",
        f"Same SOFAR schedule, only tariff differs.  "
        f"EPEX costs {yr_s4w - yr_s1w:+.0f} EUR/yr vs dag/nacht.")
    _add_arrow(ax, 0, 1, max(vals) * 1.10, yr_s4w - yr_s1w)

    ax = axes[1, 0]
    vals = [ann_s0_ep, yr_s4w, ann_lp_ep]
    lbls = ["S3b  Solar only\nEPEX (no battery)",
            "S4  Real SOFAR + wear\nEPEX + markup",
            "S3  LP + wear\nEPEX + markup"]
    _bar_panel(ax, lbls, vals, [SC["s3b"], SC["s4w"], SC["s3"]],
        "Q3: LP optimal vs real SOFAR at EPEX+markup",
        f"Real SOFAR saves {ann_s0_ep - yr_s4w:.0f} EUR/yr vs solar-only.  "
        f"LP saves an extra {yr_s4w - ann_lp_ep:.0f} EUR/yr vs real SOFAR.")
    _add_arrow(ax, 0, 1, max(vals) * 1.10, yr_s4w  - ann_s0_ep)
    _add_arrow(ax, 1, 2, max(vals) * 1.17, ann_lp_ep - yr_s4w)

    ax = axes[1, 1]
    vals = [yr_s1w, ann_lp_dn, yr_s4w, ann_lp_ep]
    lbls = ["S1  Real SOFAR\ndag/nacht  ★",
            "S2  LP\ndag/nacht",
            "S4  Real SOFAR\nEPEX + markup",
            "S3  LP\nEPEX + markup"]
    _bar_panel(ax, lbls, vals, [SC["s1w"], SC["s2"], SC["s4w"], SC["s3"]],
        "Q4: Which tariff + strategy combination is cheapest?",
        f"LP dag/nacht: {ann_lp_dn:.0f} EUR/yr  |  LP EPEX: {ann_lp_ep:.0f} EUR/yr  |  "
        f"Difference: {ann_lp_ep - ann_lp_dn:+.0f} EUR/yr")
    _add_arrow(ax, 0, 1, max(vals) * 1.07, ann_lp_dn - yr_s1w)
    _add_arrow(ax, 2, 3, max(vals) * 1.07, ann_lp_ep - yr_s4w)
    _add_arrow(ax, 1, 3, max(vals) * 1.16, ann_lp_ep - ann_lp_dn)

    plt.tight_layout(pad=1.5, h_pad=3.5, w_pad=1.0)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()

    # ── §6 Sensitivity analysis — EPEX markup breakeven ───────────────────────
    st.markdown("### Sensitivity analysis — EPEX markup breakeven")
    st.caption(
        "At what markup level does switching from dag/nacht to a dynamic EPEX tariff "
        "become profitable?  Markup = distribution, taxes & network costs on top of raw EPEX spot."
    )

    # Load pre-computed LP and rule sweeps (markup 0.05 → 0.28, daily granularity)
    _sens_lp   = pd.read_csv(os.path.join(RESULTS_DIR, "sensitivity", "markup_breakeven.csv"))
    _sens_rule = pd.read_csv(os.path.join(RESULTS_DIR, "sensitivity", "markup_breakeven_rule.csv"))

    _markup_vals = np.sort(_sens_lp["markup"].unique())

    def _ann_at_markup(df_sens, m):
        sub = df_sens[df_sens["markup"] == m].copy()
        sub.index = pd.to_datetime(sub["date"])
        return monthly_annual(sub["cost"])

    _costs_lp_yr   = np.array([_ann_at_markup(_sens_lp,   m) for m in _markup_vals])
    _costs_rule_yr = np.array([_ann_at_markup(_sens_rule, m) for m in _markup_vals])

    _ref_lp_dn    = ann_lp_dn          # LP dag/nacht annual cost (incl. wear)
    _ref_sofar_dn = ann_sf_dn + ann_sw_dn  # Real SOFAR dag/nacht (incl. wear)

    def _interp_be(x_arr, y_arr, ref):
        idx = int(np.searchsorted(y_arr, ref))
        if idx == 0: return float(x_arr[0])
        if idx >= len(x_arr): return None
        x0, x1 = x_arr[idx-1], x_arr[idx]
        y0, y1 = y_arr[idx-1], y_arr[idx]
        return float(x0 + (ref - y0) / (y1 - y0) * (x1 - x0))

    _be_lp_dn    = _interp_be(_markup_vals, _costs_lp_yr,   _ref_lp_dn)
    _be_lp_sf    = _interp_be(_markup_vals, _costs_lp_yr,   _ref_sofar_dn)
    _be_rule_sf  = _interp_be(_markup_vals, _costs_rule_yr, _ref_sofar_dn)
    _idx_cur     = int(np.argmin(np.abs(_markup_vals - MARKUP)))
    _cost_lp_cur = _costs_lp_yr[_idx_cur]
    _cost_ru_cur = _costs_rule_yr[_idx_cur]

    # ── Plot ─────────────────────────────────────────────────────────────────
    def _ax_style(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.25);   ax.spines["bottom"].set_alpha(0.25)
        ax.yaxis.grid(True, alpha=0.2, lw=0.7); ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)

    fig_s, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig_s.patch.set_facecolor("#fafafa")
    fig_s.suptitle(
        "Markup sensitivity — when does switching to EPEX become profitable?",
        fontsize=11, fontweight="bold"
    )

    # Panel 1: absolute annual costs
    ax1.set_facecolor("#fafafa")
    ax1.plot(_markup_vals, _costs_lp_yr,   color="#c0392b",   lw=2.2,
             label="LP EPEX+markup  (new optimised controller)")
    ax1.plot(_markup_vals, _costs_rule_yr, color="#e67e22",   lw=2.2, ls="--",
             label="Rule EPEX+markup  (proxy: current SOFAR on dynamic tariff)")
    ax1.axhline(_ref_lp_dn,    color="#2471a3", lw=1.8, ls="--",
                label=f"LP dag/nacht = {_ref_lp_dn:.0f} EUR/yr")
    ax1.axhline(_ref_sofar_dn, color="#7f8c8d", lw=1.8, ls=":",
                label=f"Real SOFAR dag/nacht = {_ref_sofar_dn:.0f} EUR/yr  [today]")
    ax1.axvline(MARKUP, color="#333", lw=1.2, ls=":", alpha=0.7,
                label=f"Current markup = {MARKUP:.2f} EUR/kWh")
    for _be, _col in [(_be_lp_sf, "#c0392b"), (_be_rule_sf, "#e67e22")]:
        if _be:
            ax1.axvline(_be, color=_col, lw=1.0, ls="--", alpha=0.45)
    ax1.set_ylabel("EUR / year  (incl. battery wear)", fontsize=9)
    ax1.set_title("Annual cost as a function of EPEX markup", fontsize=10, pad=6)
    ax1.legend(fontsize=8, framealpha=0.0)
    _ax_style(ax1)

    # Panel 2: saving vs Real SOFAR dag/nacht
    ax2.set_facecolor("#fafafa")
    _save_lp   = _ref_sofar_dn - _costs_lp_yr
    _save_rule = _ref_sofar_dn - _costs_rule_yr
    ax2.plot(_markup_vals, _save_lp,   color="#c0392b",  lw=2.2,
             label="LP EPEX+markup  (new controller)")
    ax2.plot(_markup_vals, _save_rule, color="#e67e22",  lw=2.2, ls="--",
             label="Rule EPEX+markup  (current SOFAR on dynamic)")
    ax2.axhline(0, color="#333", lw=0.9)
    ax2.fill_between(_markup_vals, _save_lp, 0,
                     where=_save_lp > 0, alpha=0.10, color="#27ae60",
                     label="LP EPEX cheaper than today")
    ax2.fill_between(_markup_vals, _save_rule, 0,
                     where=_save_rule > 0, alpha=0.10, color="#e67e22",
                     label="Rule EPEX cheaper than today")
    ax2.axvline(MARKUP, color="#333", lw=1.2, ls=":", alpha=0.7,
                label=f"Current markup = {MARKUP:.2f}")
    if _be_lp_sf:
        ax2.scatter([_be_lp_sf],  [0], color="#c0392b",  s=80, zorder=5,
                    label=f"LP breakeven = {_be_lp_sf:.3f} EUR/kWh")
    if _be_rule_sf:
        ax2.scatter([_be_rule_sf], [0], color="#e67e22", s=80, zorder=5,
                    label=f"Rule breakeven = {_be_rule_sf:.3f} EUR/kWh")
    ax2.set_xlabel("Markup  (EUR/kWh)", fontsize=9)
    ax2.set_ylabel("Annual saving vs Real SOFAR today  (EUR/yr)\n+ = switching is profitable",
                   fontsize=9)
    ax2.set_title("Saving from switching to EPEX  (vs real SOFAR dag/nacht today)",
                  fontsize=10, pad=6)
    ax2.legend(fontsize=8, framealpha=0.0)
    _ax_style(ax2)

    plt.tight_layout(pad=1.5, h_pad=2.0)
    st.pyplot(fig_s, use_container_width=True); plt.close()

    # Key findings
    _be_lp_str   = f"{_be_lp_sf:.3f}" if _be_lp_sf   else "not in range"
    _be_rule_str = f"{_be_rule_sf:.3f}" if _be_rule_sf else "never"
    st.info(
        f"**LP EPEX+markup** (new controller): beats Real SOFAR dag/nacht when markup ≤ "
        f"**{_be_lp_str} EUR/kWh**;  beats LP dag/nacht when markup ≤ "
        f"**{f'{_be_lp_dn:.3f}' if _be_lp_dn else 'not in range'} EUR/kWh**.  \n"
        f"**Rule EPEX+markup** (current SOFAR, just new tariff): breakeven at "
        f"**{_be_rule_str} EUR/kWh** — switching tariff *without* a new controller "
        f"{'is profitable only below that markup.' if _be_rule_sf else 'never pays off.'}  \n"
        f"At current markup {MARKUP:.2f}: LP costs **{_cost_lp_cur:.0f} EUR/yr**, "
        f"Rule costs **{_cost_ru_cur:.0f} EUR/yr** vs Real SOFAR **{_ref_sofar_dn:.0f} EUR/yr**."
    )

    st.divider()

    # ── §6b Battery parameter sensitivity ─────────────────────────────────────
    st.markdown("### Sensitivity analysis — battery capacity & power")
    st.caption(
        "What if the household had a different battery system?  "
        "Annual saving (LP EPEX + markup vs solar-only baseline) for different "
        "capacity (S_max) and power (P_max) combinations.  "
        f"Current SOFAR: S_max = {S_MAX:.0f} kWh, P_max = {P_KW:.0f} kW.  "
        "LP relaxation (binary=False) over full 2024–2026 dataset."
    )

    _sens2 = pd.read_csv(
        os.path.join(RESULTS_DIR, "sensitivity", "battery_params_sensitivity.csv"),
        parse_dates=["date"]
    )
    _sens2["saving"] = _sens2["cost_baseline"] - _sens2["cost"]

    _ann2_rows = []
    for (_s2, _p2), _grp2 in _sens2.groupby(["s_max", "p_kw"]):
        _ser2 = _grp2.set_index("date")["saving"]
        _ann2_rows.append({"s_max": _s2, "p_kw": _p2,
                           "saving_yr": monthly_annual(_ser2)})
    _df_ann2 = pd.DataFrame(_ann2_rows)
    _piv2    = _df_ann2.pivot(index="s_max", columns="p_kw", values="saving_yr")

    _S2_VALS = sorted(_piv2.index.tolist())
    _P2_VALS = sorted(_piv2.columns.tolist())

    fig_p2, (ax_hm, ax_ln) = plt.subplots(1, 2, figsize=(13, 5))
    fig_p2.patch.set_facecolor("#fafafa")
    fig_p2.suptitle(
        f"Battery parameter sensitivity  |  EPEX + markup = {MARKUP} EUR/kWh  |  "
        f"DEG = {DEG} EUR/kWh\nAnnual net saving vs solar-only baseline (no battery)",
        fontsize=10
    )

    # ── Heatmap ────────────────────────────────────────────────────────────────
    _vmin2 = min(float(_piv2.values.min()), 0)
    _vmax2 = float(_piv2.values.max()) * 1.05
    _im2   = ax_hm.imshow(_piv2.values, cmap="RdYlGn", aspect="auto",
                          vmin=_vmin2, vmax=_vmax2)
    ax_hm.set_xticks(range(len(_P2_VALS)))
    ax_hm.set_xticklabels([f"{p:.1f} kW" for p in _P2_VALS], fontsize=9)
    ax_hm.set_yticks(range(len(_S2_VALS)))
    ax_hm.set_yticklabels([f"{s:.1f} kWh" for s in _S2_VALS], fontsize=9)
    ax_hm.set_xlabel("Battery power (P_max)", fontsize=9)
    ax_hm.set_ylabel("Battery capacity (S_max)", fontsize=9)
    ax_hm.set_title("Annual saving  (EUR/yr)", fontsize=9)
    for _i2, _sm2 in enumerate(_S2_VALS):
        for _j2, _pk2 in enumerate(_P2_VALS):
            _v2 = _piv2.loc[_sm2, _pk2]
            ax_hm.text(_j2, _i2, f"{_v2:.0f}", ha="center", va="center",
                       fontsize=11, fontweight="bold",
                       color="white" if _v2 < _vmax2 * 0.30 else "black")
    _ci2 = _S2_VALS.index(S_MAX)
    _cj2 = _P2_VALS.index(P_KW)
    ax_hm.add_patch(plt.Rectangle((_cj2 - 0.5, _ci2 - 0.5), 1, 1,
                                   fill=False, edgecolor="#2471a3", lw=2.5))
    ax_hm.text(_cj2, _ci2 + 0.42, "current", ha="center", va="center",
               fontsize=7, color="#2471a3", style="italic")
    plt.colorbar(_im2, ax=ax_hm, label="EUR/yr", shrink=0.85)

    # ── Line chart ────────────────────────────────────────────────────────────
    _p2_cols = ["#e74c3c", "#e67e22", "#27ae60", "#2471a3"]
    for _pk2, _col2 in zip(_P2_VALS, _p2_cols):
        _sub2 = _df_ann2[_df_ann2["p_kw"] == _pk2].sort_values("s_max")
        ax_ln.plot(_sub2["s_max"], _sub2["saving_yr"], marker="o",
                   color=_col2, lw=1.8, ms=6, label=f"{_pk2:.1f} kW")
    ax_ln.axhline(0, color="#555", lw=0.9, ls="--", alpha=0.7, label="No saving")
    _curr2_val = float(_piv2.loc[S_MAX, P_KW])
    ax_ln.scatter([S_MAX], [_curr2_val], s=140, color="#2471a3", zorder=5,
                  label=f"Current SOFAR: {_curr2_val:.0f} EUR/yr")
    ax_ln.set_xlabel("Battery capacity S_max (kWh)", fontsize=9)
    ax_ln.set_ylabel("Annual net saving (EUR/yr)", fontsize=9)
    ax_ln.set_title("Saving vs capacity — per power level", fontsize=9)
    ax_ln.legend(fontsize=8, title="P_max", title_fontsize=8)
    ax_ln.set_facecolor("#fafafa")
    ax_ln.spines["top"].set_visible(False)
    ax_ln.spines["right"].set_visible(False)
    ax_ln.yaxis.grid(True, alpha=0.25); ax_ln.set_axisbelow(True)
    ax_ln.tick_params(labelsize=8)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig_p2, use_container_width=True); plt.close()

    _best2_idx  = np.unravel_index(np.array(_piv2.values).argmax(), _piv2.shape)
    _best2_s    = float(_piv2.index[_best2_idx[0]])
    _best2_p    = float(_piv2.columns[_best2_idx[1]])
    _best2_val  = float(_piv2.values.max())
    st.info(
        f"**Key finding:** even with the best tested hardware "
        f"(S_max = {_best2_s:.0f} kWh, P_max = {_best2_p:.0f} kW), "
        f"the annual saving is only **{_best2_val:.0f} EUR/yr** — "
        f"just **{_best2_val - _curr2_val:.0f} EUR/yr** more than the current SOFAR "
        f"({_curr2_val:.0f} EUR/yr).  \n"
        f"The bottleneck is the **EPEX price spread** and "
        f"**injection price = {PRICE_INJ} EUR/kWh**, not the hardware."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LP Оптимізація (EPEX + markup, ідеальний прогноз, 3 дні)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fc:
    st.subheader("LP Optimisation — EPEX + markup  (★ perfect forecast)")

    # ── KPI strip ─────────────────────────────────────────────────────────────
    _ann_save_ep  = ann_s0_ep - ann_lp_ep   # battery-only saving, EPEX baseline
    _ann_base_ep  = ann_s0_ep               # solar-only EPEX annual cost
    _ann_lp_ep    = ann_lp_ep              # LP EPEX annual cost

    def _kpi_md(label, value, note=""):
        note_html = f"<div style='font-size:11px;color:#888;margin-top:2px'>{note}</div>" if note else ""
        st.markdown(
            f"<div style='padding:4px 0 12px 0'>"
            f"<div style='font-size:11px;color:#555;font-weight:600;text-transform:uppercase;"
            f"letter-spacing:.04em'>{label}</div>"
            f"<div style='font-size:24px;font-weight:700;color:#1a1a2e;line-height:1.2'>{value}</div>"
            f"{note_html}</div>",
            unsafe_allow_html=True
        )

    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        _kpi_md("Battery saving / year", f"{_ann_save_ep:.0f} EUR",
                "LP EPEX vs solar-only baseline (battery-only saving)")
    with kc2:
        _kpi_md("Without battery (EPEX)", f"{_ann_base_ep:.0f} EUR/yr",
                "Solar only — upper cost bound")
    with kc3:
        _kpi_md("LP EPEX optimum", f"{_ann_lp_ep:.0f} EUR/yr",
                "★ perfect forecast + new controller")
    with kc4:
        _kpi_md("Markup", f"{MARKUP:.2f} EUR/kWh",
                f"EPEX spot + {MARKUP:.2f} = all-in tariff")

    st.caption(
        "⚠️ Saving = difference in electricity bill (LP EPEX vs solar-only, no battery). "
        "Solar self-consumption (direct use without metering) is **not included**."
    )
    st.divider()

    # ── Date selector ─────────────────────────────────────────────────────────
    _all_d26  = sorted(set(df26.index.normalize().date))
    _d26_set  = set(_all_d26)
    # need at least 3 full days from start
    _avail_fc = [d for d in _all_d26
                 if (pd.Timestamp(d) + pd.Timedelta(days=3)).date() in _d26_set]

    _sel_col, _ = st.columns([1, 2])
    with _sel_col:
        _fc_start = st.selectbox("Start date (3 days)", options=_avail_fc,
                                  index=max(0, len(_avail_fc) // 3))

    # ── Run LP ────────────────────────────────────────────────────────────────
    _fc_end_ts = pd.Timestamp(_fc_start) + pd.Timedelta(days=3)
    _sl_fc     = df[(df.index >= pd.Timestamp(_fc_start)) & (df.index < _fc_end_ts)].copy()
    _lp_fc     = run_lp(_sl_fc, use_epex=True, markup=MARKUP)

    if _lp_fc.empty:
        st.warning("No data available for this date range.")
    else:
        _ts_fc   = _lp_fc.index
        _p_fc    = _lp_fc["price"].values           # EPEX + markup
        _net_fc  = (_lp_fc["c"] - _lp_fc["d"]).values * 4  # kW (+ charge, - discharge)
        _s_fc    = _lp_fc["s"].values               # SOC kWh

        _solar_fc = (_sl_fc["sl_productie_kwh"] * 4).reindex(_ts_fc).fillna(0).values
        _load_fc  = (_sl_fc["verbruik_kwh"] * 4).reindex(_ts_fc).fillna(0).values

        _dates_fc = [(pd.Timestamp(_fc_start) + pd.Timedelta(days=i)).date() for i in range(3)]
        _sep_fc   = [pd.Timestamp(d) for d in _dates_fc[1:]]

        # ── Figure: 4 panels ──────────────────────────────────────────────────
        _GRfc = dict(alpha=0.2, lw=0.7, color="#888")

        def _fc_ax(ax):
            ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)
            ax.spines["left"].set_alpha(0.25);    ax.spines["bottom"].set_alpha(0.25)
            ax.set_facecolor("#fafafa")
            ax.yaxis.grid(True, **_GRfc); ax.set_axisbelow(True)
            ax.tick_params(labelsize=8, length=3, color="#aaa")
            for _sep in _sep_fc:
                ax.axvline(_sep, color="#bbb", lw=0.9, ls="--", alpha=0.7)

        fig_fc, axes_fc = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
        fig_fc.patch.set_facecolor("#fafafa")
        fig_fc.suptitle(
            f"LP EPEX optimisation  |  {_fc_start} → {_dates_fc[-1]}  "
            f"|  markup = {MARKUP:.2f} EUR/kWh",
            fontsize=10, color="#333"
        )

        # Panel 1 — EPEX ціна
        ax = axes_fc[0]
        ax.plot(_ts_fc, _p_fc, color="#c0392b", lw=1.4)
        ax.axhline(PRICE_DAG,   color="#2471a3", lw=0.9, ls=":", alpha=0.7,
                   label=f"DAG = {PRICE_DAG} EUR/kWh")
        ax.axhline(PRICE_NACHT, color="#7f8c8d", lw=0.9, ls=":", alpha=0.7,
                   label=f"NACHT = {PRICE_NACHT} EUR/kWh")
        # EPEX шейдинг (сірий → зелений → помаранчевий від 13:00)
        _t_pub = pd.Timestamp(_fc_start) + pd.Timedelta(hours=13)
        _t_end = _ts_fc[-1] + pd.Timedelta(minutes=15)
        ax.axvspan(_ts_fc[0], min(_t_pub, _t_end), color="#888", alpha=0.07)
        ax.axvspan(_t_pub, min(_t_pub + pd.Timedelta(hours=24), _t_end), color="#27ae60", alpha=0.11)
        if _t_pub + pd.Timedelta(hours=24) < _t_end:
            ax.axvspan(_t_pub + pd.Timedelta(hours=24), _t_end, color="#e67e22", alpha=0.17)
        for _d in _dates_fc:
            _tl = pd.Timestamp(_d) + pd.Timedelta(hours=13)
            if _ts_fc[0] < _tl < _ts_fc[-1]:
                ax.axvline(_tl, color="#27ae60", lw=0.9, ls=":", alpha=0.7)
        ax.set_ylabel("EUR/kWh", fontsize=8)
        ax.set_title("EPEX price + markup", fontsize=9, pad=4)
        ax.legend(fontsize=7, framealpha=0.0, loc="upper right", ncol=2)
        ax.set_ylim(EPEX_PRICE_YLIM)
        _fc_ax(ax)

        # Panel 2 — Сонце + Навантаження
        ax = axes_fc[1]
        ax.fill_between(_ts_fc, _solar_fc, alpha=0.55, color="#f39c12", label="Solar (kW)")
        ax.plot(_ts_fc, _load_fc, color="#2c3e50", lw=1.2, label="Load (kW)")
        ax.set_ylabel("kW", fontsize=8)
        ax.set_title("Solar generation & load", fontsize=9, pad=4)
        ax.legend(fontsize=7, framealpha=0.0)
        _fc_ax(ax)

        # Panel 3 — Заряд / розряд
        ax = axes_fc[2]
        ax.fill_between(_ts_fc, np.maximum(_net_fc, 0), color="#2ecc71",
                        alpha=0.8, label="Charge")
        ax.fill_between(_ts_fc, np.minimum(_net_fc, 0), color="#e74c3c",
                        alpha=0.8, label="Discharge")
        ax.axhline(0, color="#555", lw=0.6)
        ax.set_ylabel("kW", fontsize=8)
        ax.set_title("Battery — charge / discharge  (LP optimum)", fontsize=9, pad=4)
        ax.legend(fontsize=7, framealpha=0.0)
        _fc_ax(ax)

        # Panel 4 — SOC
        ax = axes_fc[3]
        ax.plot(_ts_fc, _s_fc, color="#2471a3", lw=1.8)
        ax.fill_between(_ts_fc, _s_fc, S_MIN, where=_s_fc >= S_MIN,
                        alpha=0.10, color="#2471a3")
        ax.axhline(S_MAX, color="#888", lw=0.9, ls=":", label=f"S_MAX = {S_MAX} kWh")
        ax.axhline(S_MIN, color="#e74c3c", lw=0.9, ls=":", label=f"S_MIN = {S_MIN} kWh")
        ax.set_ylim(-0.2, S_MAX + 0.6)
        ax.set_ylabel("kWh", fontsize=8)
        ax.set_title("State of charge (SOC)", fontsize=9, pad=4)
        ax.legend(fontsize=7, framealpha=0.0, loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
        ax.xaxis.grid(True, **_GRfc)
        _fc_ax(ax)

        plt.tight_layout(pad=1.2, h_pad=1.5)
        st.pyplot(fig_fc, use_container_width=True); plt.close()

        # ── Per-day battery activity summary ──────────────────────────────────
        _day_rows = []
        for _di, _d in enumerate(_dates_fc):
            _mask = _lp_fc.index.date == _d
            _sub  = _lp_fc[_mask]
            if _sub.empty:
                continue
            _spread = _sub["price"].max() - _sub["price"].min()
            _chg    = (_sub["c"] / 4).sum()          # kWh charged
            _dis    = (_sub["d"] / 4).sum()          # kWh discharged
            _cycled = _chg > 0.01 or _dis > 0.01
            _day_rows.append({
                "Date":         str(_d),
                "Price spread": f"{_spread:.3f} EUR/kWh",
                "Charged":      f"{_chg:.2f} kWh",
                "Discharged":   f"{_dis:.2f} kWh",
                "Battery active?": "✅ Yes" if _cycled else "❌ No (spread too small)"
            })
        if _day_rows:
            st.caption(
                "**Per-day battery decision** — the LP does not cycle the battery when "
                f"the EPEX price spread < round-trip wear cost (≈ 2 × {DEG} = {2*DEG:.2f} EUR/kWh):"
            )
            st.dataframe(pd.DataFrame(_day_rows).set_index("Date"), use_container_width=True)

        # ── Cost caption ──────────────────────────────────────────────────────
        _cost_lp_sel = (
            _lp_fc["g_in"] * _lp_fc["price"] / 4
            + _lp_fc["c"] * DEG / 4
            - _lp_fc["g_out"] * PRICE_INJ / 4
        ).sum()
        _afn_fc = (_sl_fc["afname_kwh"] - _sl_fc["ev_energie_kwh"].fillna(0)).clip(lower=0)
        _cost_nobt_sel = (
            (_afn_fc * (_sl_fc["price_eur_kwh"] + MARKUP)).sum()
            - (_sl_fc["injectie_kwh"] * PRICE_INJ).sum()
        )
        _save_sel = _cost_nobt_sel - _cost_lp_sel
        _pct_save_sel = _save_sel / _cost_nobt_sel * 100 if _cost_nobt_sel else 0
        st.caption(
            f"Selected period: LP EPEX cost = **{_cost_lp_sel:.2f} EUR**  |  "
            f"Without battery = **{_cost_nobt_sel:.2f} EUR**  |  "
            f"Saving = **{_save_sel:.2f} EUR  ({_pct_save_sel:+.1f}%)**"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Battery Calculator
# ═══════════════════════════════════════════════════════════════════════════════
with tab_calc:
    st.subheader("🔋 Battery Calculator — Custom parameters")
    st.caption(
        "Enter your battery specs and press **▶ Calculate** to run the LP optimisation "
        "on real household data (Nov 2024 – Apr 2026, solar + EPEX prices)."
    )

    def _kpi_box(label, value, note="", pct=None, pct_positive_is_good=True):
        note_html = (
            f"<div style='font-size:11px;color:#888;margin-top:2px'>{note}</div>"
            if note else ""
        )
        if pct is not None:
            good = (pct >= 0) == pct_positive_is_good
            col  = "#27ae60" if good else "#c0392b"
            pct_html = (
                f"<div style='font-size:12px;font-weight:600;color:{col};"
                f"margin-top:3px'>{pct:+.1f}% vs baseline</div>"
            )
        else:
            pct_html = ""
        st.markdown(
            f"<div style='padding:4px 0 12px 0'>"
            f"<div style='font-size:11px;color:#555;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:.04em'>{label}</div>"
            f"<div style='font-size:24px;font-weight:700;color:#1a1a2e;"
            f"line-height:1.2'>{value}</div>"
            f"{pct_html}"
            f"{note_html}</div>",
            unsafe_allow_html=True,
        )

    col_in, col_out = st.columns([1, 2], gap="large")

    # ── Left column: inputs inside a form ─────────────────────────────────────
    with col_in:
        with st.form("battery_calc_form"):
            st.markdown("##### Battery parameters")

            calc_smax = st.slider("Capacity  S_MAX (kWh)",
                                  min_value=2.5, max_value=20.0, value=5.0, step=0.5)
            calc_pkw  = st.slider("Max power  P_MAX (kW)",
                                  min_value=1.0, max_value=10.0, value=3.0, step=0.5)
            calc_cost = st.number_input("Battery cost (EUR)",
                                        min_value=500, max_value=30000,
                                        value=3000, step=100)
            calc_efc  = st.number_input("Lifetime full cycles (EFC)",
                                        min_value=500, max_value=10000,
                                        value=6000, step=100)
            calc_mu   = st.slider("EPEX markup (EUR/kWh)",
                                  min_value=0.05, max_value=0.35,
                                  value=MARKUP, step=0.01)

            submitted = st.form_submit_button("▶ Calculate", use_container_width=True)
            if submitted:
                _deg_calc = float(calc_cost) / (float(calc_efc) * float(calc_smax))
                st.session_state["calc_params"] = {
                    "s_max":    float(calc_smax),
                    "p_kw":     float(calc_pkw),
                    "deg":      round(_deg_calc, 6),
                    "markup":   float(calc_mu),
                    "cost_lbl": int(calc_cost),
                    "efc_lbl":  int(calc_efc),
                    "smax_lbl": float(calc_smax),
                    "pkw_lbl":  float(calc_pkw),
                    "deg_lbl":  _deg_calc,
                    "mu_lbl":   float(calc_mu),
                }

    # ── Right column: results ─────────────────────────────────────────────────
    with col_out:
        st.markdown("##### Results")

        if "calc_params" not in st.session_state:
            st.info("Set parameters on the left and press **▶ Calculate**.")
        else:
            _p = st.session_state["calc_params"]
            st.caption(
                f"**DEG** = {_p['cost_lbl']} / ({_p['efc_lbl']} × {_p['smax_lbl']}) = "
                f"**{_p['deg_lbl']:.4f} EUR/kWh**  ·  "
                f"S_MAX={_p['smax_lbl']} kWh  ·  P_MAX={_p['pkw_lbl']} kW  ·  "
                f"markup={_p['mu_lbl']:.2f} EUR/kWh"
            )
            with st.spinner("Running LP optimisation (~30 s)…"):
                _df_calc = _lp_custom(
                    s_max=_p["s_max"], p_kw=_p["p_kw"],
                    deg=_p["deg"],     markup=_p["markup"],
                )

            if _df_calc.empty:
                st.warning("No results — check that the data directory is accessible.")
            else:
                _ann_calc_base = monthly_annual(_df_calc["cost_baseline"])
                _ann_calc_ep   = monthly_annual(_df_calc["cost_baseline_ep"])
                _ann_calc_lp   = monthly_annual(_df_calc["cost"])
                _ann_calc_deg  = monthly_annual(_df_calc["cost_deg"])
                _ann_calc_save = _ann_calc_base - _ann_calc_lp   # solar+dag/nacht → LP EPEX+battery
                _ann_calc_gross = _ann_calc_save + _ann_calc_deg  # saving before DEG
                _ann_tariff_sw      = _ann_calc_base - _ann_calc_ep        # dag/nacht → EPEX (no battery)
                _ann_battery_gross  = _ann_calc_ep - _ann_calc_lp + _ann_calc_deg  # EPEX no bat → LP, before DEG

                _pct_gross = _ann_calc_gross / _ann_calc_base * 100 if _ann_calc_base else 0
                _pct_deg   = -_ann_calc_deg  / _ann_calc_base * 100 if _ann_calc_base else 0
                _pct_net   = _ann_calc_save  / _ann_calc_base * 100 if _ann_calc_base else 0
                _pct_lp    = (_ann_calc_lp - _ann_calc_base) / _ann_calc_base * 100 if _ann_calc_base else 0

                kk1, kk2, kk3, kk4, kk5 = st.columns(5)
                with kk1:
                    _kpi_box("Gross saving", f"{_ann_calc_gross:.0f} EUR",
                             "Before battery wear cost",
                             pct=_pct_gross, pct_positive_is_good=True)
                with kk2:
                    _kpi_box("DEG cost / year", f"− {_ann_calc_deg:.0f} EUR",
                             "Battery wear (already deducted from net saving)",
                             pct=_pct_deg, pct_positive_is_good=False)
                with kk3:
                    _kpi_box("Net saving ★", f"{_ann_calc_save:.0f} EUR",
                             f"= {_ann_calc_gross:.0f} − {_ann_calc_deg:.0f}  |  solar+dag/nacht → LP EPEX",
                             pct=_pct_net, pct_positive_is_good=True)
                with kk4:
                    _kpi_box("Without battery (dag/nacht)", f"{_ann_calc_base:.0f} EUR/yr",
                             "Solar only, fixed dag/nacht tariff — no battery")
                with kk5:
                    _kpi_box("With LP battery (EPEX)", f"{_ann_calc_lp:.0f} EUR/yr",
                             "LP optimised on dynamic EPEX tariff",
                             pct=_pct_lp, pct_positive_is_good=False)

                st.divider()

                # ── Saving decomposition ──────────────────────────────────────
                st.markdown("**Saving decomposition**")
                st.caption(
                    f"Tariff switch (dag/nacht → EPEX, no battery): **{_ann_tariff_sw:+.0f} EUR/yr**  ·  "
                    f"Battery gross (EPEX + LP, before DEG): **{_ann_battery_gross:+.0f} EUR/yr**  ·  "
                    f"= **Gross {_ann_calc_gross:.0f} EUR**  −  DEG {_ann_calc_deg:.0f} EUR  "
                    f"= **Net {_ann_calc_save:.0f} EUR/yr**"
                )

                # ── Monthly saving bar chart ───────────────────────────────────
                _df_calc2 = _df_calc.copy()
                _df_calc2.index = pd.to_datetime(_df_calc2.index)
                _monthly_save = (
                    (_df_calc2["cost_baseline"] - _df_calc2["cost"])
                    .resample("ME").sum(min_count=1)
                )
                _monthly_deg = _df_calc2["cost_deg"].resample("ME").sum(min_count=1)
                _monthly_net = _monthly_save - _monthly_deg

                fig_calc, ax_calc = plt.subplots(figsize=(9, 4))
                _xi   = range(len(_monthly_save))
                _xlbl = _monthly_save.index.strftime("%b %Y")
                ax_calc.bar([i - 0.2 for i in _xi], _monthly_save, 0.38,
                            color="#27ae60", alpha=0.85, label="Gross saving")
                ax_calc.bar([i + 0.2 for i in _xi], _monthly_net,  0.38,
                            color="#2471a3", alpha=0.85, label="Net saving (after DEG)")
                ax_calc.axhline(0, color="#333", lw=0.8)
                ax_calc.set_xticks(list(_xi))
                ax_calc.set_xticklabels(list(_xlbl), rotation=45, ha="right", fontsize=7)
                ax_calc.set_ylabel("EUR / month")
                ax_calc.set_title(
                    f"Monthly saving — S_MAX={_p['smax_lbl']} kWh, "
                    f"P_MAX={_p['pkw_lbl']} kW, DEG={_p['deg_lbl']:.4f} EUR/kWh",
                    fontsize=9,
                )
                ax_calc.legend(fontsize=8)
                ax_calc.yaxis.grid(True, alpha=0.3); ax_calc.set_axisbelow(True)
                plt.tight_layout()
                st.pyplot(fig_calc, use_container_width=True)
                plt.close()

                st.caption(
                    f"Parameters: S_MAX={_p['smax_lbl']} kWh · P_MAX={_p['pkw_lbl']} kW · "
                    f"DEG={_p['deg_lbl']:.4f} EUR/kWh · markup={_p['mu_lbl']:.2f} EUR/kWh"
                )

                st.divider()

                # ── Payback estimate ───────────────────────────────────────────
                st.markdown("**Payback estimate** *(ideal scenario)*")
                _battery_cost = float(_p["cost_lbl"])
                if _ann_calc_save <= 0:
                    st.warning(
                        "Net saving ≤ 0 EUR/yr — battery does not pay back under these parameters."
                    )
                else:
                    _payback_yr = _battery_cost / _ann_calc_save
                    if _payback_yr <= 15:
                        _pb_color = "#27ae60"
                    elif _payback_yr <= 25:
                        _pb_color = "#e67e22"
                    else:
                        _pb_color = "#c0392b"
                    st.markdown(
                        f"<div style='padding:10px 14px;border-radius:8px;"
                        f"border-left:4px solid {_pb_color};background:#f8f9fa'>"
                        f"<span style='font-size:14px;font-weight:600'>Battery cost</span> "
                        f"<span style='font-size:22px;font-weight:700;color:{_pb_color}'>"
                        f" {_battery_cost:,.0f} EUR</span>"
                        f" ÷ <span style='font-size:14px;font-weight:600'>net saving</span> "
                        f"<span style='font-size:22px;font-weight:700;color:{_pb_color}'>"
                        f"{_ann_calc_save:.0f} EUR/yr</span>"
                        f" = <span style='font-size:26px;font-weight:800;color:{_pb_color}'>"
                        f" {_payback_yr:.1f} years</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "⚠ **Ideal scenario assumptions:** LP uses perfect day-ahead price foresight · "
                        "battery capacity assumed constant over lifetime · "
                        "based on historical EPEX prices (Nov 2024 – Apr 2026) · "
                        "installation / inverter costs not included."
                    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ML Forecast LP
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ml:
    st.subheader("ML Forecast LP — day-ahead schedule with ML consumption + solar forecasts")
    st.caption(
        "Window: selected date 13:00 → next day 24:00 (35 h · 140 slots · 15 min). "
        "LP optimises using ML forecasts; costs evaluated at actual measured load & solar. "
        "Only dates with recorded SOFAR SOC at 13:00 are shown (Jan / Apr 2026 — out-of-sample)."
    )

    if not _ml_avail_dates:
        st.warning("No dates with SOC data found.")
        st.stop()

    # ── Controls ──────────────────────────────────────────────────────────────
    _ml_c1, _ml_c2, _ml_c3, _ml_c4, _ml_c5, _ml_c6 = st.columns(6)
    with _ml_c1:
        _ml_date = st.selectbox("Date (13:00 start)", options=_ml_avail_dates,
                                index=len(_ml_avail_dates) // 2)
    with _ml_c2:
        _ml_smax = st.slider("S_MAX (kWh)", 2.5, 10.0, S_MAX, 0.5, key="ml_smax")
    with _ml_c3:
        _ml_pkw  = st.slider("P_MAX (kW)",  1.0,  5.0, float(P_KW), 0.5, key="ml_pkw")
    with _ml_c4:
        _ml_smin = st.slider("S_MIN (kWh)", 0.0,  2.0, S_MIN, 0.1, key="ml_smin")
    with _ml_c5:
        _ml_deg  = st.slider("DEG (EUR/kWh)", 0.0, 0.20, DEG, 0.01, key="ml_deg",
                             format="%.2f")
    with _ml_c6:
        _ml_mu   = st.slider("Markup (EUR/kWh)", 0.05, 0.35, MARKUP, 0.01, key="ml_mu",
                             format="%.2f")

    st.divider()

    # ── Build 35-hour window ──────────────────────────────────────────────────
    _ml_start = pd.Timestamp(_ml_date).replace(hour=13)
    _ml_end   = _ml_start + pd.Timedelta(hours=35) - pd.Timedelta(minutes=15)
    _ml_win   = df.loc[_ml_start : _ml_end].copy()
    _ml_pmax  = _ml_pkw / 4.0   # kW → kWh/slot

    if len(_ml_win) < 100:
        st.warning(f"Not enough data for window {_ml_start} → {_ml_end}. Try another date.")
        st.stop()

    # Initial SOC: percentage from SOFAR → kWh (scaled to current S_MAX slider)
    _soc_pct = float(_ml_win["soc_begin"].dropna().iloc[0]) \
               if _ml_win["soc_begin"].notna().any() else 50.0
    _ml_s0   = min(max(_soc_pct / 100.0 * _ml_smax, _ml_smin), _ml_smax)

    # ── LP helper: run on window with ML forecasts, evaluate at actual ────────
    def _run_ml_lp(use_epex, markup_val):
        p      = (_ml_win["price_eur_kwh"] + markup_val).values if use_epex \
                 else _ml_win["tarief_price"].values
        l_fc   = _ml_win["verbruik_fc"].fillna(_ml_win["verbruik_kwh"]).values
        sol_fc = np.maximum(_ml_win["sl_productie_forecast"].fillna(0).values, 0)
        res    = optimize_day(p, l_fc, _ml_smax, _ml_pmax, ETA_C, ETA_D, _ml_s0,
                              cyclic=False, binary=True, deg_cost=_ml_deg,
                              S_min=_ml_smin, solar=sol_fc, price_inj=PRICE_INJ)
        l_act   = _ml_win["verbruik_kwh"].values
        sol_act = _ml_win["sl_productie_kwh"].values
        net     = l_act + res["c"] - res["d"] - sol_act
        g_in    = np.maximum(net, 0)
        g_out   = np.maximum(-net, 0)
        cost    = (float(np.dot(p, g_in))
                   - PRICE_INJ * float(g_out.sum())
                   + res["cost_degradation"])
        return pd.DataFrame({
            "price": p, "c": res["c"], "d": res["d"], "s": res["s"],
            "g_in": g_in, "g_out": g_out,
            "l_fc": l_fc, "l_act": l_act, "sol_fc": sol_fc, "sol_act": sol_act,
        }, index=_ml_win.index), cost

    _res_dn, _cost_dn = _run_ml_lp(use_epex=False, markup_val=0.0)
    _res_ep, _cost_ep = _run_ml_lp(use_epex=True,  markup_val=_ml_mu)

    # SOFAR actual cost on this window (dag/nacht, actual flows + wear)
    _afn_ml     = (_ml_win["afname_kwh"] - _ml_win["ev_energie_kwh"].fillna(0)).clip(lower=0)
    _cost_sofar = (float((_afn_ml * _ml_win["tarief_price"]).sum())
                   - PRICE_INJ * float(_ml_win["injectie_kwh"].sum())
                   + float(_ml_win["bat_laden_kwh_kw"].clip(0, 0.75).sum()) * DEG)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    _kc1, _kc2, _kc3, _kc4 = st.columns(4)
    def _ml_kpi(col, label, value_str, note, delta=None):
        with col:
            if delta is not None:
                _vc = "#27ae60" if delta >= 0 else "#c0392b"
                _vh = f"<span style='color:{_vc}'>{value_str}</span>"
            else:
                _vh = value_str
            st.markdown(
                f"<div style='padding:4px 0 10px 0'>"
                f"<div style='font-size:11px;color:#555;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:.04em'>{label}</div>"
                f"<div style='font-size:20px;font-weight:700;color:#1a1a2e;"
                f"line-height:1.2'>{_vh}</div>"
                f"<div style='font-size:11px;color:#888;margin-top:2px'>{note}</div>"
                f"</div>", unsafe_allow_html=True)

    _ml_kpi(_kc1, "SOFAR cost (35 h)",
            f"{_cost_sofar:.2f} EUR", "actual system · dag/nacht tariff")
    _ml_kpi(_kc2, "LP dag/nacht saving vs SOFAR",
            f"{_cost_sofar - _cost_dn:+.2f} EUR",
            "same tariff · ML forecasts", delta=_cost_sofar - _cost_dn)
    _ml_kpi(_kc3, "LP EPEX saving vs SOFAR",
            f"{_cost_sofar - _cost_ep:+.2f} EUR",
            f"EPEX + markup {_ml_mu:.2f} · ML forecasts", delta=_cost_sofar - _cost_ep)
    _ml_kpi(_kc4, "Initial SOC",
            f"{_soc_pct:.0f}%  ({_ml_s0:.2f} kWh)",
            f"SOFAR reading at {_ml_start.strftime('%d/%m %H:%M')}")

    st.divider()

    # ── Chart helper ──────────────────────────────────────────────────────────
    _GR_ml = dict(alpha=0.2, lw=0.7, color="#888")

    def _ml_ax(ax, ts):
        ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.25);    ax.spines["bottom"].set_alpha(0.25)
        ax.set_facecolor("#fafafa")
        ax.yaxis.grid(True, **_GR_ml);  ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, length=3, color="#aaa")
        # Midnight separator
        for _sep in pd.date_range(ts[0].normalize() + pd.Timedelta(days=1),
                                   ts[-1], freq="D"):
            if ts[0] < _sep < ts[-1]:
                ax.axvline(_sep, color="#bbb", lw=0.8, ls="--", alpha=0.6)

    # Shared y-axis limits for price and charge/discharge panels
    _all_prices  = pd.concat([_res_dn["price"], _res_ep["price"]])
    _p_margin    = (_all_prices.max() - _all_prices.min()) * 0.08 + 0.01
    _price_ylim  = (_all_prices.min() - _p_margin, _all_prices.max() + _p_margin)

    _dn_net = (_res_dn["c"] - _res_dn["d"]) * 4
    _ep_net = (_res_ep["c"] - _res_ep["d"]) * 4
    _bat_abs = max(abs(_dn_net).max(), abs(_ep_net).max(), 0.1)
    _bat_ylim = (-_bat_abs * 1.12, _bat_abs * 1.12)

    def _make_ml_fig(df_res, use_epex):
        fig, axes = plt.subplots(4, 1, figsize=(6.5, 11), sharex=True)
        fig.patch.set_facecolor("#fafafa")
        ts = df_res.index

        # Panel 1 — price
        ax = axes[0]
        clr = "#c0392b" if use_epex else "#2471a3"
        ax.plot(ts, df_res["price"], color=clr, lw=1.4)
        if not use_epex:
            ax.axhline(PRICE_DAG,   color="#2471a3", lw=0.8, ls=":", alpha=0.7,
                       label=f"Dag {PRICE_DAG}")
            ax.axhline(PRICE_NACHT, color="#95a5a6", lw=0.8, ls=":", alpha=0.7,
                       label=f"Nacht {PRICE_NACHT}")
            ax.legend(fontsize=7, framealpha=0.0)
        ax.set_ylim(_price_ylim)
        ax.set_ylabel("EUR/kWh", fontsize=8)
        ax.set_title("Dag/nacht tariff" if not use_epex else f"EPEX + markup {_ml_mu:.2f}",
                     fontsize=9, pad=4)
        _ml_ax(ax, ts)

        # Panel 2 — forecast vs actual
        ax = axes[1]
        ax.plot(ts, df_res["l_act"] * 4, color="#2c3e50", lw=1.3, label="Load actual")
        ax.plot(ts, df_res["l_fc"]  * 4, color="#2c3e50", lw=1.0, ls="--",
                alpha=0.5, label="Load forecast")
        ax.fill_between(ts, df_res["sol_act"] * 4, alpha=0.55,
                        color="#f39c12", label="Solar actual")
        ax.fill_between(ts, df_res["sol_fc"]  * 4, alpha=0.25,
                        color="#e67e22", label="Solar forecast")
        ax.set_ylabel("kW", fontsize=8)
        ax.set_title("Forecast vs actual — load & solar", fontsize=9, pad=4)
        ax.legend(fontsize=7, framealpha=0.0, ncol=2)
        _ml_ax(ax, ts)

        # Panel 3 — LP charge/discharge
        net_kw = (df_res["c"] - df_res["d"]) * 4
        ax = axes[2]
        ax.fill_between(ts, np.maximum(net_kw, 0), color="#2ecc71", alpha=0.8, label="Charge")
        ax.fill_between(ts, np.minimum(net_kw, 0), color="#e74c3c", alpha=0.8,
                        label="Discharge")
        ax.axhline(0, color="#555", lw=0.6)
        ax.set_ylim(_bat_ylim)
        ax.set_ylabel("kW", fontsize=8)
        ax.set_title("Battery schedule (LP · ML forecasts)", fontsize=9, pad=4)
        ax.legend(fontsize=7, framealpha=0.0)
        _ml_ax(ax, ts)

        # Panel 4 — SOC
        s = df_res["s"].values
        ax = axes[3]
        ax.plot(ts, s, color="#2471a3", lw=1.8)
        ax.fill_between(ts, s, _ml_smin, where=s >= _ml_smin, alpha=0.10, color="#2471a3")
        ax.axhline(_ml_smax, color="#888",    lw=0.9, ls=":", label=f"S_MAX = {_ml_smax} kWh")
        ax.axhline(_ml_smin, color="#e74c3c", lw=0.9, ls=":", label=f"S_MIN = {_ml_smin} kWh")
        ax.set_ylim(-0.2, _ml_smax + 0.8)
        ax.set_ylabel("kWh", fontsize=8)
        ax.set_title("State of charge (SOC)", fontsize=9, pad=4)
        ax.legend(fontsize=7, framealpha=0.0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
        ax.xaxis.grid(True, **_GR_ml)
        _ml_ax(ax, ts)

        plt.tight_layout(pad=1.0, h_pad=1.2)
        return fig

    # ── Side-by-side columns ──────────────────────────────────────────────────
    _col_dn, _col_ep = st.columns(2)

    with _col_dn:
        st.markdown(
            f"#### Dag/nacht  ·  cost = **{_cost_dn:.2f} EUR**  "
            f"·  vs SOFAR = **{_cost_sofar - _cost_dn:+.2f} EUR**"
        )
        st.pyplot(_make_ml_fig(_res_dn, use_epex=False), use_container_width=True)
        plt.close()

    with _col_ep:
        st.markdown(
            f"#### EPEX + {_ml_mu:.2f}  ·  cost = **{_cost_ep:.2f} EUR**  "
            f"·  vs SOFAR = **{_cost_sofar - _cost_ep:+.2f} EUR**"
        )
        st.pyplot(_make_ml_fig(_res_ep, use_epex=True), use_container_width=True)
        plt.close()

