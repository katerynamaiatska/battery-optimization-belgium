from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, lpSum, value, PULP_CBC_CMD
import numpy as np
import pandas as pd

# ── LP optimisation ──────────────────────────────────────────
def optimize_day(prices, load, S_max, P_max, eta_c, eta_d, S_init, cyclic=True, binary=False, deg_cost=0.0, S_min=0.0):
    """
    Find the optimal charge/discharge schedule for one day using LP.

    Args:
        prices   : array of hourly prices (EUR/kWh), length T
        load     : array of hourly consumption (H0 relative units), length T
        S_max    : battery capacity (kWh)
        P_max    : max charge/discharge power per hour (kW)
        eta_c    : charge efficiency — fraction of grid energy stored in battery (e.g. 0.975)
        eta_d    : discharge efficiency — fraction of battery energy delivered to load (e.g. 0.975)
        S_init   : state of charge at start of day (kWh)
        cyclic   : if True, force s[T-1] >= S_init (prevents free-energy trick)
        binary   : if True, add binary variable to forbid simultaneous charge+discharge (MILP, slower)
        deg_cost : degradation cost per kWh charged (EUR/kWh); penalises battery wear in objective
        S_min    : minimum allowed SOC (kWh); protects battery from deep discharge (e.g. 0.1 * S_max)

    Returns dict: status, cost, cost_electricity, cost_degradation,
                  charge schedule (c), discharge (d), SOC (s), s_final
    """

    T = len(prices)
    prob = LpProblem("battery_day", LpMinimize)

    c = [LpVariable(f"c_{t}", lowBound=0, upBound=P_max) for t in range(T)]
    d = [LpVariable(f"d_{t}", lowBound=0, upBound=P_max) for t in range(T)]
    s = [LpVariable(f"s_{t}", lowBound=S_min, upBound=S_max) for t in range(T)]

    if binary:
        z = [LpVariable(f"z_{t}", cat='Binary') for t in range(T)]
        for t in range(T):
            prob += c[t] <= P_max * z[t]       # if z=1 → charge
            prob += d[t] <= P_max * (1 - z[t])  # if z=0 → discharge

    # Objective: electricity cost + battery degradation cost per kWh charged
    prob += lpSum(prices[t] * (load[t] + c[t] - d[t]) + deg_cost * c[t] for t in range(T))

    # Constraints
    # Energy balance: use multiplication by 1/eta_d (PuLP does not support LpVariable / scalar)
    inv_eta_d = 1.0 / eta_d
    prob += s[0] == S_init + eta_c * c[0] - inv_eta_d * d[0]
    for t in range(1, T):
        prob += s[t] == s[t-1] + eta_c * c[t] - inv_eta_d * d[t]

    # do not sell to the electricity grid
    for t in range(T):
        prob += d[t] <= load[t]

    # so as not to use the initial charge "as free" 
    if cyclic:
        prob += s[T-1] >= S_init

    prob.solve(PULP_CBC_CMD(msg=0))

    c_vals = [value(c[t]) or 0.0 for t in range(T)]
    d_vals = [value(d[t]) or 0.0 for t in range(T)]

    # Separate the two cost components so callers can report them independently
    cost_degradation = deg_cost * sum(c_vals)           # EUR "consumed" from battery lifetime
    cost_electricity = value(prob.objective) - cost_degradation  # actual grid bill

    return {
        "status":           LpStatus[prob.status],
        "cost":             value(prob.objective),  # total = electricity + degradation
        "cost_electricity": cost_electricity,        # what appears on the electricity bill
        "cost_degradation": cost_degradation,        # battery wear cost for this day
        "c":                c_vals,
        "d":                d_vals,
        "s":                [value(s[t]) for t in range(T)],
        "s_final":          value(s[T-1])            # carried over to next day
    }


def threshold_strategy(prices, load, S_max, P_max, eta_c, eta_d, S_init, cyclic=True, deg_cost=0.0, S_min=0.0):
    """
    Simple rule-based strategy: charge when price < daily mean, discharge otherwise.
    Much faster than LP — useful as a benchmark.

    Args: same as optimize_day (S_min: minimum SOC below which discharge is blocked)
    Returns dict: cost, cost_electricity, cost_degradation,
                  charge schedule (c), discharge (d), SOC (s), s_final
    """

    T = len(prices)
    threshold = np.mean(prices)  # average price as a threshold

    s = S_init
    total_cost = 0
    c_val, d_val, s_val = [], [], []

    for t in range(T):
        # Negative price: grid pays YOU to consume — always charge, never discharge
        # Positive price: charge only when cheaper than daily average (adjusted for wear)
        if prices[t] < 0 or prices[t] + deg_cost < threshold:
            charge = min(P_max, S_max - s)
            discharge = 0
        else:
            charge = 0
            discharge = min(P_max, (s - S_min) * eta_d, load[t])  # limited by energy above S_min

        if cyclic and t == T - 1:
            s_after = s + eta_c * charge - discharge / eta_d
            if s_after < S_init:
                # SOC would end below S_init — stop discharging and charge instead
                # (keeping both charge and discharge > 0 would be simultaneous, which is invalid)
                discharge = 0
                needed_soc = S_init - s - eta_c * charge
                if needed_soc > 0:
                    charge = min(charge + needed_soc / eta_c, P_max)

        s = s + eta_c * charge - discharge / eta_d
        total_cost += prices[t] * (load[t] + charge - discharge)
        
        c_val.append(charge)
        d_val.append(discharge)
        s_val.append(s)
    
    # total_cost already contains only electricity (deg_cost affects the decision, not this sum)
    cost_electricity = total_cost
    cost_degradation = deg_cost * sum(c_val)  # battery wear cost for this day

    return {
        "cost":             cost_electricity + cost_degradation,  # total, comparable with LP
        "cost_electricity": cost_electricity,                      # actual grid bill
        "cost_degradation": cost_degradation,                      # battery wear cost
        "c":                c_val,
        "d":                d_val,
        "s":                s_val,
        "s_final":          s
    }


def backtest(prices_df, load_df, S_max, P_max, eta_c, eta_d, S_init, strategy='lp', binary=True, deg_cost=0.0, S_min=0.0):
    """
    Run day-by-day backtesting over the full prices/load dataset.
    Battery state (s_final) carries over from one day to the next.

    Args:
        prices_df : DataFrame with DatetimeIndex (tz-aware) and column 'price_eur_kwh'
        load_df   : DataFrame with DatetimeIndex (no tz) and column 'consumption'
        strategy  : 'lp' (optimal, slow) or 'threshold' (rule-based, fast)
        deg_cost  : degradation cost per kWh charged passed to optimize_day (EUR/kWh)
        others    : same battery parameters as optimize_day

    Returns: DataFrame indexed by date with columns
             cost, cost_electricity, cost_degradation, cost_baseline,
             saving_electricity, saving_net, s_final, year, month
    """
    results = []
    s = S_init

    # Remove timezone so index aligns with load (which has no tz)
    prices_naive = prices_df.copy()
    prices_naive.index = prices_naive.index.tz_localize(None)
    p_by_day = prices_naive['price_eur_kwh'].groupby(pd.DatetimeIndex(prices_naive.index).date)
    l_by_day = load_df['consumption'].groupby(pd.DatetimeIndex(load_df.index).date)

    common_dates = sorted(set(p_by_day.groups) & set(l_by_day.groups))

    for date in common_dates:
        p = p_by_day.get_group(date).values
        l = l_by_day.get_group(date).values

        T_day = len(p)
        if T_day not in [23, 24, 25]:  # protection against real data errors
            continue

        # Load profile is always 24h (no DST); align to price hours
        if len(l) < T_day:
            l = np.append(l, l[-1:] * (T_day - len(l)))  # fall back: repeat last hour
        elif len(l) > T_day:
            l = l[:T_day]                                  # spring forward: trim last hour

        cost_base = float(np.dot(p, l))

        if strategy == 'lp':
            res = optimize_day(p, l, S_max, P_max, eta_c, eta_d, s, cyclic=False, binary=binary, deg_cost=deg_cost, S_min=S_min)
        else:
            res = threshold_strategy(p, l, S_max, P_max, eta_c, eta_d, s, cyclic=False, deg_cost=deg_cost, S_min=S_min)

        results.append({
            'date':             date,
            'cost':             res['cost'],             # total: electricity + degradation
            'cost_electricity': res['cost_electricity'], # actual grid bill (no battery wear)
            'cost_degradation': res['cost_degradation'], # battery wear cost for this day
            'cost_baseline':    cost_base,               # grid bill without any battery
            # saving_electricity: how much less we paid the grid vs no-battery scenario
            'saving_electricity': cost_base - res['cost_electricity'],
            # saving_net: real profit after subtracting battery wear from electricity saving
            'saving_net':       cost_base - res['cost'],
            's_final':          res['s_final'],
            'year':             date.year,
            'month':            date.month,
        })

        s = res['s_final']

    return pd.DataFrame(results).set_index('date')
