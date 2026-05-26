from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, lpSum, value, PULP_CBC_CMD
import numpy as np
import pandas as pd

# ── LP optimisation ──────────────────────────────────────────
def optimize_day(prices, load, S_max, P_max, eta, S_init, cyclic=True):
    """
    Find the optimal charge/discharge schedule for one day using LP.

    Args:
        prices  : array of hourly prices (EUR/kWh), length T
        load    : array of hourly consumption (H0 relative units), length T
        S_max   : battery capacity (kWh)
        P_max   : max charge/discharge power per hour (kW)
        eta     : one-way efficiency (e.g. 0.95)
        S_init  : state of charge at start of day (kWh)
        cyclic  : if True, force s[T-1] >= S_init (prevents free-energy trick)

    Returns dict: status, cost, charge schedule (c), discharge (d), SOC (s), s_final
    """

    T = len(prices)
    prob = LpProblem("battery_day", LpMinimize)

    c = [LpVariable(f"c_{t}", lowBound=0, upBound=P_max) for t in range(T)]
    d = [LpVariable(f"d_{t}", lowBound=0, upBound=P_max) for t in range(T)]
    s = [LpVariable(f"s_{t}", lowBound=0, upBound=S_max) for t in range(T)]

    # Target function
    prob += lpSum(prices[t] * (load[t] + c[t] - d[t]) for t in range(T))

    # Constraints
    # Energy balance
    prob += s[0] == S_init + eta * c[0] - d[0]
    for t in range(1, T):
        prob += s[t] == s[t-1] + eta * c[t] - d[t]

    # do not sell to the electricity grid
    for t in range(T):
        prob += d[t] <= load[t]

    # so as not to use the initial charge "as free" 
    if cyclic:
        prob += s[T-1] >= S_init

    prob.solve(PULP_CBC_CMD(msg=0))

    return {
        "status":    LpStatus[prob.status],
        "cost":      value(prob.objective),
        "c":         [value(c[t]) for t in range(T)],
        "d":         [value(d[t]) for t in range(T)],
        "s":         [value(s[t]) for t in range(T)],
        "s_final":   value(s[T-1])   # for a chain between days
    }


def threshold_strategy(prices, load, S_max, P_max, eta, S_init, cyclic=True):
    """
    Simple rule-based strategy: charge when price < daily mean, discharge otherwise.
    Much faster than LP — useful as a benchmark.

    Args: same as optimize_day (no cyclic parameter)
    Returns dict: cost, charge schedule (c), discharge (d), SOC (s), s_final
    """

    T = len(prices)
    threshold = np.mean(prices)  # average price as a threshold
    
    s = S_init
    total_cost = 0
    c_val, d_val, s_val = [], [], []
    
    for t in range(T):
        if prices[t] < threshold:
            # Cheap hour - charge
            charge = min(P_max, S_max - s)  # no more than a free space
            discharge = 0
        else:
            # Dear hour - discharging
            charge = 0
            discharge = min(P_max, s, load[t])  # no more than is in the battery and consumption
        
        if cyclic and t == T - 1:
            needed = S_init - s - eta * charge + discharge
            if needed > 0:
                charge += min(needed, P_max, S_max - s)

        s = s + eta * charge - discharge
        total_cost += prices[t] * (load[t] + charge - discharge)
        
        c_val.append(charge)
        d_val.append(discharge)
        s_val.append(s)
    
    return {
        "cost": total_cost,
        "c": c_val,
        "d": d_val,
        "s": s_val,
        "s_final": s
    }


def backtest(prices_df, load_df, S_max, P_max, eta, S_init, strategy='lp'):
    """
    Run day-by-day backtesting over the full prices/load dataset.
    Battery state (s_final) carries over from one day to the next.

    Args:
        prices_df : DataFrame with DatetimeIndex (tz-aware) and column 'price_eur_kwh'
        load_df   : DataFrame with DatetimeIndex (no tz) and column 'consumption'
        strategy  : 'lp' (optimal, slow) or 'threshold' (rule-based, fast)
        others    : same battery parameters as optimize_day

    Returns: DataFrame indexed by date with columns
             cost, cost_baseline, saving, s_final, year, month
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

        # Skip incomplete days (e.g. DST transitions give 23 or 25 hours)
        if len(p) != 24 or len(l) != 24:
            continue

        cost_base = float(np.dot(p, l))

        if strategy == 'lp':
            res = optimize_day(p, l, S_max, P_max, eta, s, cyclic=False)
        else:
            res = threshold_strategy(p, l, S_max, P_max, eta, s, cyclic=False)

        results.append({
            'date':          date,
            'cost':          res['cost'],
            'cost_baseline': cost_base,
            'saving':        cost_base - res['cost'],
            's_final':       res['s_final'],
            'year':          date.year,
            'month':         date.month,
        })

        s = res['s_final']

    return pd.DataFrame(results).set_index('date')
