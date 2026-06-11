from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, lpSum, value, PULP_CBC_CMD
import numpy as np
import pandas as pd

# ── Solar input (sl_productie_kwh) — known limitation ────────────────────────
# sl_productie_kwh is derived from the AC energy balance:
#   solar = verbruik + injectie + bat_laden - bat_ontladen - afname, clipped >= 0
# This measures *delivered* solar (what the household actually received),
# NOT panel-side generation. Two known sources of underestimation:
#
#   1. EV-charging days (~141 days/year, ~23 kWh/year deficit, ~2% of annual solar)
#      afname includes EV charging; verbruik (OwnDev) does not.
#      Formula misses EV as load → underestimates solar on those days.
#      Root cause: ev_energie_kwh is a per-session lump sum, not per 15-min slot,
#      so it cannot be included in the 15-min energy balance without creating
#      fake solar peaks at EV session timestamps.
#
#   2. SOFAR freeze events (89 periods on 56 days, 17-month dataset)
#      When the SOFAR ME3000SP inverter freezes, it stops all energy flows
#      (bat_laden, bat_ontladen, injectie all go to 0). Formula gives solar=0.
#      This is CORRECT for the LP: no energy was usable by the household.
#      The panel-side logger shows non-zero generation, but the inverter
#      did not deliver it — so the LP cannot act on it.
#
# Conclusion: sl_productie_kwh is the right input for LP optimisation.
# The ~28% gap vs the panel logger is explained by (1) + (2) above.
# No correction is applied. Validated in 01_eda_real_load.ipynb §SOFAR-freeze.
# ─────────────────────────────────────────────────────────────────────────────

# ── LP optimisation ──────────────────────────────────────────
def optimize_day(prices, load, S_max, P_max, eta_c, eta_d, S_init, cyclic=True, binary=False, deg_cost=0.0, S_min=0.0, S_final_min=None, solar=None, price_inj=None):
    """
    Find the optimal charge/discharge schedule for one day using LP.

    Args:
        prices    : array of buy prices (EUR/kWh), length T (one entry per time slot)
        load      : array of consumption (kWh/slot), length T
        S_max     : battery capacity (kWh)
        P_max     : max energy per slot (kWh/slot) — for 15-min data: kW × 0.25 (e.g. 3 kW → 0.75)
        eta_c     : charge efficiency (e.g. 0.95)
        eta_d     : discharge efficiency (e.g. 0.95)
        S_init    : state of charge at start of day (kWh)
        cyclic    : if True, force s[T-1] >= S_init (prevents free-energy trick)
        binary    : if True, forbid simultaneous charge+discharge (MILP, slower)
        deg_cost  : degradation cost per kWh charged (EUR/kWh)
        S_min     : minimum allowed SOC (kWh)
        S_final_min: optional minimum end-of-day SOC (kWh)
        solar     : array of hourly solar production (kWh), length T; None → zeros
        price_inj : array or scalar injection price (EUR/kWh); None → same as prices

    Returns dict: status, cost, cost_electricity, cost_degradation,
                  c, d, s, s_final, g_in, g_out
    """

    T = len(prices)

    # Part A: defaults for solar and injection price
    sol = np.zeros(T) if solar is None else np.asarray(solar, dtype=float)
    p_inj = np.asarray(prices, dtype=float) if price_inj is None else np.broadcast_to(price_inj, T)

    prob = LpProblem("battery_day", LpMinimize)

    # Part B: LP variables
    c   = [LpVariable(f"c_{t}",   lowBound=0, upBound=P_max) for t in range(T)]
    d   = [LpVariable(f"d_{t}",   lowBound=0, upBound=P_max) for t in range(T)]
    s   = [LpVariable(f"s_{t}",   lowBound=S_min, upBound=S_max) for t in range(T)]
    # Upper bounds prevent unbounded LP when buy_price < inject_price (e.g. negative EPEX + PRICE_INJ=0)
    g_in  = [LpVariable(f"gin_{t}",  lowBound=0, upBound=float(load[t]) + P_max) for t in range(T)]
    g_out = [LpVariable(f"gout_{t}", lowBound=0, upBound=float(sol[t])  + P_max) for t in range(T)]

    if binary:
        z = [LpVariable(f"z_{t}", cat='Binary') for t in range(T)]
        for t in range(T):
            prob += c[t] <= P_max * z[t]
            prob += d[t] <= P_max * (1 - z[t])

    # Part C: objective — minimise net grid cost minus injection revenue plus battery wear
    prob += lpSum(
        prices[t] * g_in[t] - p_inj[t] * g_out[t] + deg_cost * c[t]
        for t in range(T)
    )

    # Part D: constraints
    inv_eta_d = 1.0 / eta_d

    # Battery SOC balance (unchanged — solar does not affect the battery directly)
    prob += s[0] == S_init + eta_c * c[0] - inv_eta_d * d[0]
    for t in range(1, T):
        prob += s[t] == s[t-1] + eta_c * c[t] - inv_eta_d * d[t]

    # Grid energy balance: g_in − g_out = load + charge − discharge − solar
    # d[t] is energy delivered to load (after efficiency losses), so it directly offsets load
    for t in range(T):
        prob += g_in[t] - g_out[t] == load[t] + c[t] - d[t] - sol[t]

    if cyclic:
        prob += s[T-1] >= S_init

    if S_final_min is not None:
        prob += s[T-1] >= S_final_min

    prob.solve(PULP_CBC_CMD(msg=False))

    c_vals   = [value(c[t])   or 0.0 for t in range(T)]
    d_vals   = [value(d[t])   or 0.0 for t in range(T)]
    gin_vals = [value(g_in[t])  or 0.0 for t in range(T)]
    gout_vals= [value(g_out[t]) or 0.0 for t in range(T)]

    cost_degradation = deg_cost * sum(c_vals)
    cost_electricity = sum(prices[t] * gin_vals[t] - p_inj[t] * gout_vals[t] for t in range(T))

    return {
        "status":           LpStatus[prob.status],
        "cost":             cost_electricity + cost_degradation,
        "cost_electricity": cost_electricity,
        "cost_degradation": cost_degradation,
        "c":                c_vals,
        "d":                d_vals,
        "s":                [value(s[t]) for t in range(T)],  # type: ignore[misc]
        "s_final":          value(s[T-1]),                    # type: ignore[return-value]
        "g_in":             gin_vals,
        "g_out":            gout_vals,
    }


def threshold_strategy(prices, load, S_max, P_max, eta_c, eta_d, S_init, cyclic=True, deg_cost=0.0, S_min=0.0, solar=None, price_inj=None):
    """
    Simple rule-based strategy: charge when price < daily mean, discharge otherwise.
    Much faster than LP — useful as a benchmark.

    Args: same as optimize_day
    Returns dict: cost, cost_electricity, cost_degradation,
                  c, d, s, s_final, g_in, g_out
    """

    T = len(prices)
    sol   = np.zeros(T) if solar is None else np.asarray(solar, dtype=float)
    p_inj = np.asarray(prices, dtype=float) if price_inj is None else np.broadcast_to(price_inj, T)

    threshold = np.mean(prices)

    s = S_init
    total_cost = 0
    c_val, d_val, s_val, gin_val, gout_val = [], [], [], [], []

    for t in range(T):
        # Negative price: grid pays YOU to consume — always charge, never discharge
        # Positive price: charge only when cheaper than daily average (adjusted for wear)
        if prices[t] < 0 or prices[t] + deg_cost < threshold:
            charge = min(P_max, S_max - s)
            discharge = 0
        else:
            charge = 0
            discharge = min(P_max, (s - S_min) * eta_d, load[t])

        if cyclic and t == T - 1:
            s_after = s + eta_c * charge - discharge / eta_d
            if s_after < S_init:
                discharge = 0
                needed_soc = S_init - s - eta_c * charge
                if needed_soc > 0:
                    charge = min(charge + needed_soc / eta_c, P_max)

        s = s + eta_c * charge - discharge / eta_d

        # Grid balance: positive = import, negative = export
        net = load[t] + charge - discharge - sol[t]
        g_in  = max(net, 0.0)
        g_out = max(-net, 0.0)

        total_cost += prices[t] * g_in - p_inj[t] * g_out

        c_val.append(charge)
        d_val.append(discharge)
        s_val.append(s)
        gin_val.append(g_in)
        gout_val.append(g_out)

    cost_electricity = total_cost
    cost_degradation = deg_cost * sum(c_val)

    return {
        "cost":             cost_electricity + cost_degradation,
        "cost_electricity": cost_electricity,
        "cost_degradation": cost_degradation,
        "c":                c_val,
        "d":                d_val,
        "s":                s_val,
        "s_final":          s,
        "g_in":             gin_val,
        "g_out":            gout_val,
    }


def backtest(prices_df, load_df, S_max, P_max, eta_c, eta_d, S_init, strategy='lp', binary=True, deg_cost=0.0, S_min=0.0, solar_df=None, price_inj=None):
    """
    Run day-by-day backtesting over the full prices/load/solar dataset.
    Battery state (s_final) carries over from one day to the next.

    Args:
        prices_df : DataFrame with DatetimeIndex (15-min) and column 'price_eur_kwh'
        load_df   : DataFrame with DatetimeIndex (15-min) and column 'consumption' (kWh/15min)
        solar_df  : DataFrame with DatetimeIndex (15-min) and column 'solar' (kWh/15min); None → no solar
        price_inj : injection price (EUR/kWh scalar); None → same as buy price
        strategy  : 'lp' (optimal, slow) or 'threshold' (rule-based, fast)
        deg_cost  : degradation cost per kWh charged (EUR/kWh)
        P_max     : max energy per 15-min slot (kWh) — e.g. 3 kW battery → P_max = 0.75
        others    : same battery parameters as optimize_day

    Returns: DataFrame indexed by date with columns
             cost, cost_electricity, cost_degradation,
             cost_baseline (solar but no battery),
             saving_electricity, saving_net, s_final, year, month
    """
    results = []
    s = S_init

    p_by_day = prices_df['price_eur_kwh'].groupby(pd.DatetimeIndex(prices_df.index).date)
    l_by_day = load_df['consumption'].groupby(pd.DatetimeIndex(load_df.index).date)

    has_solar = solar_df is not None
    sol_by_day = solar_df['solar'].groupby(pd.DatetimeIndex(solar_df.index).date) if has_solar else None

    common_dates = sorted(set(p_by_day.groups) & set(l_by_day.groups))
    if sol_by_day is not None:
        common_dates = sorted(set(common_dates) & set(sol_by_day.groups))

    def _align(arr: np.ndarray, T: int) -> np.ndarray:
        if len(arr) < T:
            return np.append(arr, np.repeat(arr[-1], T - len(arr)))
        return arr[:T]

    for date in common_dates:
        p = p_by_day.get_group(date).values
        l = l_by_day.get_group(date).values

        T_day = len(p)
        # 15-min resolution: normal=96, DST spring-forward=92, DST fall-back=100
        if T_day not in [92, 96, 100]:
            continue

        l = _align(l, T_day)

        if sol_by_day is not None:
            sol = _align(sol_by_day.get_group(date).values, T_day)
        else:
            sol = np.zeros(T_day)

        # Injection price for this day
        p_inj = price_inj if price_inj is not None else None

        # Baseline cost: solar panels but NO battery
        # → buy what solar doesn't cover, sell solar surplus
        gin_base  = np.maximum(l - sol, 0)
        gout_base = np.maximum(sol - l, 0)
        p_inj_arr = p if p_inj is None else np.full(T_day, p_inj)
        cost_base = float(np.dot(p, gin_base) - np.dot(p_inj_arr, gout_base))

        if strategy == 'lp':
            res = optimize_day(p, l, S_max, P_max, eta_c, eta_d, s,
                               cyclic=False, binary=binary, deg_cost=deg_cost,
                               S_min=S_min, solar=sol, price_inj=p_inj)
        else:
            res = threshold_strategy(p, l, S_max, P_max, eta_c, eta_d, s,
                                     cyclic=False, deg_cost=deg_cost,
                                     S_min=S_min, solar=sol, price_inj=p_inj)

        results.append({
            'date':               date,
            'cost':               res['cost'],
            'cost_electricity':   res['cost_electricity'],
            'cost_degradation':   res['cost_degradation'],
            'cost_baseline':      cost_base,         # solar but no battery
            'saving_electricity': cost_base - res['cost_electricity'],
            'saving_net':         cost_base - res['cost'],
            's_final':            res['s_final'],
            'year':               date.year,
            'month':              date.month,
        })

        s = res['s_final']

    return pd.DataFrame(results).set_index('date')
