from pulp import *

# ── LP task ──────────────────────────────────────────────────
def optimize_day(prices, load, S_max, P_max, eta, S_init, cyclic=True):

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
