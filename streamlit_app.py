import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "optimization" / "src"))
from battery_utils import optimize_day, backtest

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Battery Optimizer", page_icon="🔋", layout="wide")

# ── Pre-curated scenarios ─────────────────────────────────────────────────────
SCENARIOS: dict[str, str] = {
    "⚡ Crisis peak — Aug 2022  (max price spread, best arbitrage)": "2022-08-28",
    "☀️ Solar summer — Aug 2024  (34 h negative prices)":           "2024-08-22",
    "🌸 Spring 2025 — balanced  (high spread + solar)":             "2025-05-10",
    "💨 Winter winds — Dec 2023  (34 h negative, wind surplus)":    "2023-12-23",
    "📊 Normal winter — Jan 2024  (typical baseline behaviour)":    "2024-01-15",
}

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(__file__).parent / "Data" / "prepared"
    prices = pd.read_csv(base / "prices_clean.csv", index_col=0, parse_dates=True)
    prices.index = pd.DatetimeIndex(prices.index).tz_convert("Europe/Brussels")
    load_df = pd.read_csv(base / "load_clean.csv", index_col=0, parse_dates=True)
    return prices, load_df

prices, load_df = load_data()

# Strip timezone once — used in both tabs
prices_tz = prices.copy()
prices_tz.index = pd.DatetimeIndex(prices_tz.index).tz_localize(None)

# Valid date range for Live Planner (need 72h from 13:00)
_max_lp = (prices_tz.index.max() - pd.Timedelta(hours=85)).date()   # 72 + 13
_min_lp = pd.Timestamp("2025-01-01").date()

# ── Sidebar (shared battery params + per-tab selectors) ───────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("📊 Tab 1 — Scenario")
    scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()))
    start_date    = SCENARIOS[scenario_name]
    st.caption(f"3-day window starting {start_date}")

    st.divider()

    st.subheader("🏠 Tab 2 — Live Planner")
    _default_lp = min(pd.Timestamp("2025-05-09").date(), _max_lp)
    live_date = st.date_input(
        "Start date (13:00 fixed)",
        value=_default_lp,
        min_value=_min_lp,
        max_value=_max_lp,
    )
    st.caption(f"{live_date} 13:00 → +72 h  (35 h known + 37 h forecast)")

    st.divider()
    st.subheader("🔋 Battery")
    S_max     = st.slider("Capacity (kWh)", 5.0, 20.0, 10.0, 0.5)
    P_max     = st.slider("Max power (kW)",  1.0, 10.0,  5.0, 0.5)
    S_min_pct = st.slider("Min SOC (%)", 0, 30, 10, 5)
    S_min     = S_min_pct / 100 * S_max
    S_init_pct = st.slider("Initial SOC (%)", 0, 100, 50, 5)
    S_init    = S_init_pct / 100 * S_max
    eta_rt    = st.slider("Round-trip efficiency (%)", 85, 99, 95, 1) / 100
    eta_c = eta_d = eta_rt ** 0.5
    deg_cost  = st.number_input(
        "Degradation cost (EUR/kWh)", 0.00, 0.10, 0.02, 0.005, format="%.3f"
    )

    st.divider()
    st.subheader("🏠 Household")
    annual_kwh = st.number_input(
        "Annual consumption (kWh/yr)", 1000, 15000, 3500, 500
    )

# ── Title + tabs ──────────────────────────────────────────────────────────────
st.title("🔋 Home Battery Optimizer — Belgium")
tab1, tab2 = st.tabs(["📊 Scenario Demo", "🏠 Live Planner"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  –  Scenario Demo  (backtest vs 72 h forecast, side-by-side)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.caption(
        "72-hour LP optimisation · "
        "**Left**: day-by-day (no forecast) · "
        "**Right**: 72-hour window (perfect forecast) · "
        "EPEX day-ahead prices (historical demo)"
    )

    HORIZON = 72
    N_DAYS  = 3

    start_ts = pd.Timestamp(start_date)
    end_ts   = start_ts + pd.Timedelta(days=N_DAYS)

    prices_win = prices_tz[(prices_tz.index >= start_ts) & (prices_tz.index < end_ts)]
    load_win   = load_df[(load_df.index   >= start_ts) & (load_df.index   < end_ts)]

    p_all  = prices_win["price_eur_kwh"].to_numpy(dtype=float)[:HORIZON]
    ts_all = prices_win.index[:HORIZON]
    l_raw  = load_win["consumption"].to_numpy(dtype=float)[:HORIZON] * (annual_kwh / 3500)

    T = len(p_all)
    if len(l_raw) < T:
        l_all = np.append(l_raw, np.repeat(l_raw[-1], T - len(l_raw)))
    elif len(l_raw) > T:
        l_all = l_raw[:T]
    else:
        l_all = l_raw

    if T == 0:
        st.error(f"No price data for {start_date}.")
        st.stop()

    # Strategy A: day-by-day backtest
    prices_3d = prices[(prices.index >= start_ts.tz_localize("Europe/Brussels")) &
                       (prices.index <  end_ts.tz_localize("Europe/Brussels"))]
    load_3d   = load_df[(load_df.index >= start_ts) & (load_df.index < end_ts)].copy()
    load_3d["consumption"] *= annual_kwh / 3500

    with st.spinner("Running day-by-day backtest..."):
        bt = backtest(prices_3d, load_3d, S_max, P_max, eta_c, eta_d, S_init,
                      strategy="lp", binary=False, deg_cost=deg_cost, S_min=S_min)

    c_bt, d_bt, s_bt = [], [], []
    s = S_init
    for date in bt.index:
        p = prices_win[prices_win.index.date == date]["price_eur_kwh"].to_numpy(dtype=float)
        l = load_3d[load_3d.index.date == date]["consumption"].to_numpy(dtype=float)
        res = optimize_day(p, l, S_max, P_max, eta_c, eta_d, s,
                           cyclic=False, binary=False, deg_cost=deg_cost, S_min=S_min)
        c_bt.extend(res["c"])
        d_bt.extend(res["d"])
        s_bt.append(s)
        s_bt.extend(res["s"])
        s = res["s_final"]

    # Strategy B: 72-hour single LP
    with st.spinner("Running 72-hour forecast optimisation..."):
        res_fc = optimize_day(p_all, l_all, S_max, P_max, eta_c, eta_d, S_init,
                              cyclic=False, binary=False, deg_cost=deg_cost, S_min=S_min)

    cost_base = float(np.dot(p_all, l_all))
    saving_bt = cost_base - float(bt["cost"].sum())
    saving_fc = cost_base - float(res_fc["cost"])
    extra_fc  = saving_fc - saving_bt

    saving_fc_day, s_fc_eod = [], []
    for i in range(N_DAYS):
        h   = slice(i * 24, (i + 1) * 24)
        c_d = np.array(res_fc["c"][h])
        d_d = np.array(res_fc["d"][h])
        cost_e = float(np.dot(p_all[h], l_all[h] + c_d - d_d))
        saving_fc_day.append(bt["cost_baseline"].iloc[i] - cost_e - deg_cost * float(c_d.sum()))
        s_fc_eod.append(float(res_fc["s"][(i + 1) * 24 - 1]))

    # KPI — row 1: costs
    cost_bt_total = float(bt["cost"].sum())
    cost_fc_total = float(res_fc["cost"])

    st.markdown("**What you pay over 3 days (EUR)**")
    ca, cb, cc = st.columns(3)
    ca.metric("No battery (baseline)",       f"€ {cost_base:.3f}",
              help="Grid electricity bill with no battery at all")
    cb.metric("With battery — no forecast",  f"€ {cost_bt_total:.3f}",
              delta=f"{-saving_bt:+.3f} € vs baseline", delta_color="off",
              help="Day-by-day LP, no knowledge of tomorrow's prices. "
                   "Negative cost = battery earns money from negative prices.")
    cc.metric("With battery — 72h forecast", f"€ {cost_fc_total:.3f}",
              delta=f"{-saving_fc:+.3f} € vs baseline", delta_color="off",
              help="72-hour LP, optimizer sees all 3 days at once. "
                   "Negative cost = battery earns money from negative prices.")

    # KPI — row 2: savings
    st.markdown("**How much you save compared to no battery (EUR)**")
    cd, ce, cf = st.columns(3)
    cd.metric("Saving — no forecast",    f"€ {saving_bt:.3f}",
              delta=f"{saving_bt / cost_base * 100:+.1f}% of baseline" if cost_base > 0.01 else None,
              help="Electricity saving minus battery degradation cost (day-by-day LP)")
    ce.metric("Saving — 72h forecast",   f"€ {saving_fc:.3f}",
              delta=f"{saving_fc / cost_base * 100:+.1f}% of baseline" if cost_base > 0.01 else None,
              help="Electricity saving minus battery degradation cost (72h LP)")
    cf.metric("Extra gain from forecast", f"€ {extra_fc:.3f}",
              delta=f"{extra_fc / saving_bt * 100:+.1f}%" if saving_bt > 0.01 else None,
              help="How much more the 72h forecast earns compared to day-by-day")

    st.divider()

    # ── Charts
    x           = list(range(T))
    tick_pos    = list(range(0, T, 6))
    tick_labels = [ts_all[i].strftime("%a\n%H:%M") for i in tick_pos if i < len(ts_all)]

    x_soc_bt: list[float] = []
    for i in range(N_DAYS):
        x_soc_bt.extend([i * 24 - 0.5] + list(range(i * 24, i * 24 + 24)))
    x_soc_fc  = [-0.5] + x
    s_fc_plot  = [S_init] + [float(v) for v in res_fc["s"]]
    bar_colors = ["#e74c3c" if p < 0 else "#4a90d9" for p in p_all]

    def day_lines(ax: Axes) -> None:
        for i in range(1, N_DAYS):
            ax.axvline(i * 24 - 0.5, color="gray", lw=1, ls="--", alpha=0.5)

    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig_price, ax_p = plt.subplots(figsize=(14, 2.8))
    ax_p.bar(x, p_all, color=bar_colors, alpha=0.85)
    ax_p.axhline(float(np.mean(p_all)), color="orange", lw=1.5, ls="--",
                 label=f"3-day avg: {float(np.mean(p_all)):.4f} EUR/kWh")
    day_lines(ax_p)
    for i in range(N_DAYS):
        ax_p.text(i * 24 + 0.5, ax_p.get_ylim()[1] * 0.88,
                  (pd.Timestamp(start_date) + pd.Timedelta(days=i)).strftime("%a %d %b"),
                  fontsize=9, color="gray")
    ax_p.set_ylabel("Price (EUR/kWh)")
    ax_p.set_title(f"EPEX prices — {scenario_name}  (red = negative price)")
    ax_p.set_xticks(tick_pos[:len(tick_labels)])
    ax_p.set_xticklabels(tick_labels, fontsize=8)
    ax_p.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_price)
    plt.close()

    fig2, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col", sharey="row")

    for col, (label, c_v, d_v, s_v, x_s) in enumerate([
        (f"Backtest  (saving € {saving_bt:.3f})\nEach day sees only its own 24 h",
         c_bt, d_bt, s_bt, x_soc_bt),
        (f"72-hour forecast  (saving € {saving_fc:.3f})\nOptimizer sees all 72 h at once",
         res_fc["c"], res_fc["d"], s_fc_plot, x_soc_fc),
    ]):
        day_lines(axes[0, col])
        day_lines(axes[1, col])

        axes[0, col].bar(x, c_v, color="#2ecc71", alpha=0.85, label="Charge")
        axes[0, col].bar(x, [-d for d in d_v], color="#e74c3c", alpha=0.85, label="Discharge")
        axes[0, col].axhline(0, color="black", lw=0.6)
        axes[0, col].set_ylabel("Power (kW)")
        axes[0, col].set_title(label, fontsize=10)
        axes[0, col].legend(fontsize=8)

        axes[1, col].plot(x_s, s_v, color="orange", lw=2, marker="o", ms=3, label="SOC")
        axes[1, col].axhline(S_min, color="#e74c3c", lw=1.2, ls="--",
                             label=f"S_min ({S_min_pct}%)")
        axes[1, col].axhline(S_max, color="gray", lw=1, ls=":",
                             label=f"S_max ({S_max} kWh)")
        axes[1, col].set_ylim(0, S_max * 1.15)
        axes[1, col].set_ylabel("SOC (kWh)")
        axes[1, col].set_xlabel("Time")
        axes[1, col].set_xticks(tick_pos[:len(tick_labels)])
        axes[1, col].set_xticklabels(tick_labels, fontsize=8)
        axes[1, col].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.divider()

    # Per-day comparison table
    st.subheader("📋 Per-day comparison")
    rows = []
    for i, date in enumerate(bt.index):
        rows.append({
            "Date":               str(date),
            "Baseline (EUR)":     f"{bt['cost_baseline'].iloc[i]:.3f}",
            "Saving — backtest":  f"{bt['saving_net'].iloc[i]:.3f}",
            "s_final backtest":   f"{bt['s_final'].iloc[i]:.2f} kWh",
            "Saving — forecast":  f"{saving_fc_day[i]:.3f}",
            "s_final forecast":   f"{s_fc_eod[i]:.2f} kWh",
        })
    totals = {
        "Date":               "TOTAL",
        "Baseline (EUR)":     f"{bt['cost_baseline'].sum():.3f}",
        "Saving — backtest":  f"{saving_bt:.3f}",
        "s_final backtest":   "—",
        "Saving — forecast":  f"{saving_fc:.3f}",
        "s_final forecast":   "—",
    }
    df_table = pd.concat([pd.DataFrame(rows), pd.DataFrame([totals])], ignore_index=True)
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    st.caption(
        "**Saving — backtest**: day-by-day LP, no knowledge of tomorrow's prices.  "
        "**Saving — forecast**: 72-hour LP, optimizer sees the full window at once.  "
        "Negative per-day forecast saving = optimizer 'invested' in charging that day "
        "to discharge at better prices later."
    )

    st.divider()

    # Hourly schedule table
    st.subheader("🕐 Hourly schedule")
    st.caption("BT = day-by-day backtest  ·  FC = 72h forecast  ·  SOC shown at end of each hour")
    rows_h = []
    for t in range(T):
        day_i  = t // 24
        hour_i = t % 24
        soc_bt_t = s_bt[day_i * 25 + hour_i + 1]
        rows_h.append({
            "Time":            ts_all[t].strftime("%a %d/%m %H:%M") if t < len(ts_all) else f"h{t}",
            "Price (€/kWh)":   round(float(p_all[t]), 4),
            "Load (kWh)":      round(float(l_all[t]), 3),
            "Charge BT (kW)":  round(float(c_bt[t]), 3),
            "Disch. BT (kW)":  round(float(d_bt[t]), 3),
            "SOC BT (kWh)":    round(float(soc_bt_t), 2),
            "Charge FC (kW)":  round(float(res_fc["c"][t]), 3),
            "Disch. FC (kW)":  round(float(res_fc["d"][t]), 3),
            "SOC FC (kWh)":    round(float(s_fc_plot[t + 1]), 2),
        })
    st.dataframe(pd.DataFrame(rows_h), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  –  Live Planner  (13:00 start · 35 h known + 37 h forecast)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    START_HOUR = 13
    SPLIT_H    = 48 - START_HOUR   # 35 h of known EPEX prices (today 13–24 + all of tomorrow)
    FORECAST_H = 72 - SPLIT_H      # 37 h forecast (day-after-tomorrow)

    lp_start = pd.Timestamp(f"{live_date} {START_HOUR:02d}:00")
    lp_end   = lp_start + pd.Timedelta(hours=72)

    prices_lp = prices_tz[(prices_tz.index >= lp_start) & (prices_tz.index < lp_end)]
    load_lp   = load_df[(load_df.index   >= lp_start) & (load_df.index   < lp_end)]

    p_lp  = prices_lp["price_eur_kwh"].to_numpy(dtype=float)[:72]
    ts_lp = prices_lp.index[:72]
    l_raw_lp = load_lp["consumption"].to_numpy(dtype=float)[:72] * (annual_kwh / 3500)

    T_lp = len(p_lp)
    if len(l_raw_lp) < T_lp:
        l_lp = np.append(l_raw_lp, np.repeat(l_raw_lp[-1], T_lp - len(l_raw_lp)))
    elif len(l_raw_lp) > T_lp:
        l_lp = l_raw_lp[:T_lp]
    else:
        l_lp = l_raw_lp

    if T_lp < 72:
        st.warning(f"Only {T_lp} h of price data available from {lp_start}. Choose an earlier date.")
        st.stop()

    st.caption(
        f"**Window**: {lp_start.strftime('%a %d %b %Y %H:%M')} → "
        f"{lp_end.strftime('%a %d %b %Y %H:%M')}  ·  "
        f"**Known EPEX prices**: {SPLIT_H} h (today 13:00–24:00 + all of tomorrow)  ·  "
        f"**Forecast**: {FORECAST_H} h (day-after-tomorrow 00:00–12:00)  ·  "
        "Historical prices used as forecast proxy"
    )

    with st.spinner("Running 72-hour optimisation..."):
        res_lp = optimize_day(p_lp, l_lp, S_max, P_max, eta_c, eta_d, S_init,
                              cyclic=False, binary=False, deg_cost=deg_cost, S_min=S_min)

    cost_base_lp = float(np.dot(p_lp, l_lp))
    saving_lp    = cost_base_lp - float(res_lp["cost"])

    # KPI
    st.markdown("**What you pay over 72 hours (EUR)**")
    lca, lcb, lcc = st.columns(3)
    lca.metric("No battery (baseline)",      f"€ {cost_base_lp:.3f}",
               help="Grid bill for 72 h without any battery")
    lcb.metric("With battery + forecast",    f"€ {float(res_lp['cost']):.3f}",
               delta=f"{-saving_lp:+.3f} € vs baseline", delta_color="off",
               help="72h LP using known EPEX + forecast prices. "
                    "Negative cost = battery earns money from negative prices.")
    lcc.metric("Saving over 72 h",           f"€ {saving_lp:.3f}",
               delta=f"{saving_lp / cost_base_lp * 100:+.1f}% of baseline" if cost_base_lp > 0.01 else None)

    st.divider()

    # ── Combined 3-panel figure: price / charge-discharge / SOC ─────────────
    # All panels share the same x-axis → perfect vertical alignment.
    # x=0 corresponds to 13:00 (the first bar); SOC starts at x=0 with S_init.
    x_lp        = list(range(T_lp))
    tick_pos_lp = list(range(0, T_lp, 8))   # every 8 h to avoid crowding
    tick_lp     = [ts_lp[i].strftime("%a\n%H:%M") for i in tick_pos_lp if i < len(ts_lp)]

    bar_clr_lp = ["#c0392b" if p < 0 else "#4a90d9" for p in p_lp]

    # SOC convention: s_lp_plot[0] = S_init at t=0 (13:00), then end-of-hour values
    # Shift SOC one step so x=t shows the SOC *at the start* of hour t
    s_lp_eoh   = [float(v) for v in res_lp["s"]]           # 72 end-of-hour values
    soc_x      = list(range(T_lp + 1))                      # 0..72
    soc_y      = [S_init] + s_lp_eoh                        # 73 values: start + 72 ends
    # Clamp last tick to T_lp so the axis doesn't extend beyond 72
    soc_x_plot = soc_x[:-1]                                  # 0..71 → aligns with bars
    soc_y_plot = soc_y[:-1]                                  # drop last point (outside window)
    # Add the final SOC as a separate marker at the right edge
    soc_x_final = T_lp - 0.5
    soc_y_final = soc_y[-1]

    from matplotlib.transforms import blended_transform_factory

    fig_lp, (ax_pr, ax_cd, ax_soc) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5, 1.5]},
    )
    fig_lp.suptitle(
        f"72h window — {lp_start.strftime('%a %d %b %Y %H:%M')} → "
        f"{lp_end.strftime('%a %d %b %H:%M')}  ·  saving € {saving_lp:.3f}",
        fontsize=11,
    )

    # — shared decorations: split line + forecast shading
    for ax in (ax_pr, ax_cd, ax_soc):
        ax.axvspan(SPLIT_H - 0.5, T_lp - 0.5, color="#f0f0f0", alpha=0.6, zorder=0)
        ax.axvline(SPLIT_H - 0.5, color="black", lw=1.5, ls="--")

    # — panel 1: prices
    bars_lp = ax_pr.bar(x_lp, p_lp, color=bar_clr_lp, alpha=0.85, edgecolor="none", zorder=2)
    for i in range(SPLIT_H, T_lp):
        bars_lp[i].set_hatch("///")
        bars_lp[i].set_edgecolor("white")
        bars_lp[i].set_linewidth(0.5)

    trans_pr = blended_transform_factory(ax_pr.transData, ax_pr.transAxes)
    ax_pr.text(0.5, 0.95, f"◀ Known EPEX ({SPLIT_H} h)",
               transform=ax_pr.transAxes, fontsize=9, color="#2471a3",
               fontweight="bold", va="top", ha="center")
    ax_pr.text((SPLIT_H + T_lp) / 2 / T_lp, 0.95, f"Forecast ({FORECAST_H} h) ▶",
               transform=ax_pr.transAxes, fontsize=9, color="#566573",
               fontweight="bold", va="top", ha="center")
    for i, ts in enumerate(ts_lp):
        if ts.hour == 0:
            ax_pr.axvline(i - 0.5, color="gray", lw=0.8, ls="--", alpha=0.3)
            ax_pr.text(i + 0.4, 0.04, ts.strftime("%a %d %b"),
                       transform=trans_pr, fontsize=8, color="gray")
    ax_pr.set_ylabel("Price (EUR/kWh)")
    ax_pr.axhline(0, color="black", lw=0.5)

    # — panel 2: charge / discharge
    trans_cd = blended_transform_factory(ax_cd.transData, ax_cd.transAxes)
    ax_cd.bar(x_lp, res_lp["c"], color="#2ecc71", alpha=0.85, label="Charge", zorder=2)
    ax_cd.bar(x_lp, [-d for d in res_lp["d"]], color="#e74c3c", alpha=0.85,
              label="Discharge", zorder=2)
    ax_cd.axhline(0, color="black", lw=0.6)
    ax_cd.set_ylabel("Power (kW)")
    ax_cd.legend(fontsize=8)
    for i, ts in enumerate(ts_lp):
        if ts.hour == 0:
            ax_cd.text(i + 0.4, 0.04, ts.strftime("%a %d %b"),
                       transform=trans_cd, fontsize=8, color="gray")

    # — panel 3: SOC
    ax_soc.plot(soc_x_plot, soc_y_plot, color="orange", lw=2, marker="o", ms=3, label="SOC")
    ax_soc.plot(soc_x_final, soc_y_final, color="orange", marker="D", ms=5)  # end marker
    ax_soc.axhline(S_min, color="#e74c3c", lw=1.2, ls="--", label=f"S_min ({S_min_pct}%)")
    ax_soc.axhline(S_max, color="gray",    lw=1,   ls=":",  label=f"S_max ({S_max} kWh)")
    ax_soc.set_ylim(0, S_max * 1.15)
    ax_soc.set_ylabel("SOC (kWh)")
    ax_soc.set_xlabel("Time")
    ax_soc.legend(fontsize=8)

    # Shared x-axis ticks (only shown on bottom panel)
    ax_soc.set_xticks(tick_pos_lp[:len(tick_lp)])
    ax_soc.set_xticklabels(tick_lp, fontsize=8)
    ax_soc.set_xlim(-0.5, T_lp - 0.5)   # tight range: no axis before 13:00

    plt.tight_layout()
    st.pyplot(fig_lp)
    plt.close()

    st.divider()

    # Hourly schedule table
    st.subheader("🕐 Hourly schedule")
    s_lp_all = [S_init] + [float(v) for v in res_lp["s"]]
    rows_lp = []
    for t in range(T_lp):
        rows_lp.append({
            "Time":          ts_lp[t].strftime("%a %d/%m %H:%M") if t < len(ts_lp) else f"h{t}",
            "Type":          "Known" if t < SPLIT_H else "Forecast",
            "Price (€/kWh)": round(float(p_lp[t]), 4),
            "Load (kWh)":    round(float(l_lp[t]), 3),
            "Charge (kW)":   round(float(res_lp["c"][t]), 3),
            "Disch. (kW)":   round(float(res_lp["d"][t]), 3),
            "SOC (kWh)":     round(float(s_lp_all[t + 1]), 2),
        })
    st.dataframe(pd.DataFrame(rows_lp), use_container_width=True, hide_index=True)
