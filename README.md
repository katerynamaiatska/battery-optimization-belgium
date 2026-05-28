# Home Battery Optimization under Dynamic Electricity Tariffs
**Data Science Internship — Belgium | 2026**

---

## Project idea

Belgian consumers on a dynamic electricity contract pay a price that changes every hour, based on EPEX SPOT day-ahead market prices. Most of them don't know when to charge or discharge their home battery to actually save money.

This project builds an optimization system that computes the ideal charge/discharge schedule for a home battery, uses real historical prices to quantify savings over 4 years, and provides an interactive demo for households considering a battery purchase.

A comparable tool exists for the Netherlands ([energie.theoxygent.nl](https://energie.theoxygent.nl)) — this project asks the same question for Belgium.

---

## What is implemented

### 1. Data collection (`01_data_collection.ipynb`)
- Hourly EPEX SPOT day-ahead prices for Belgium via ENTSO-E (2022–2025)
- Hourly weather data for Antwerp via Open-Meteo (temperature, wind speed, solar radiation, cloud cover)
- Synthetic H0 household load profile (3 500 kWh/year baseline) scaled to any household size
- Cleaned and aligned outputs saved to `Data/prepared/`

### 2. Exploratory Data Analysis (`02_eda.ipynb`)
- Price distributions per year, month, and hour of day
- Negative price analysis: frequency, duration, seasonal patterns
- Price volatility: daily spread, peak/off-peak ratio
- Household consumption patterns and load-price correlation
- Identification of representative scenarios for demonstration

### 3. LP Optimization (`03_optimization.ipynb` + `optimization/src/battery_utils.py`)

**Core LP model — `optimize_day()`**

Minimize total cost over a T-hour horizon:

```
Minimize  Σ [ p(t) × (load(t) + c(t) − d(t)) + deg_cost × c(t) ]
```

Subject to:
- SOC balance: `s(t) = s(t−1) + η_c · c(t) − d(t) / η_d`
- Capacity limits: `S_min ≤ s(t) ≤ S_max`
- Power limits: `0 ≤ c(t) ≤ P_max`,  `0 ≤ d(t) ≤ P_max`
- No grid export: `d(t) ≤ load(t)`
- Optional cyclic constraint: `s(T) ≥ s(0)` (prevents free-energy trick)
- Optional MILP mode: binary variable prevents simultaneous charge + discharge

**Rule-based benchmark — `threshold_strategy()`**

Charge when `price(t) < daily_mean − deg_cost`, discharge otherwise. Fast heuristic used to verify LP advantage and for sensitivity sweeps.

**Multi-day backtest — `backtest()`**

Runs either strategy day by day over the full 2022–2025 dataset. Battery state-of-charge carries over between days. Tracks electricity cost, degradation cost, and net saving separately.

**Section 3b** compares plain LP vs MILP (binary constraint): the LP without binary already achieves optimal or near-optimal results at a fraction of the solve time.

### 4. Backtest results (2022–2025, 10 kWh / 5 kW / η_rt 95%)

| Strategy | Annual saving (incl. 21% VAT) |
|---|---|
| LP optimisation | ~120 EUR/year |
| Threshold rule | ~68 EUR/year |
| LP advantage over rule | ~52 EUR/year (+76%) |

2022 (energy crisis) produced 274 EUR/yr; normal years 2023–2025 averaged 103 EUR/yr. The crisis year inflates the multi-year average.

### 5. Sensitivity analysis (`03_optimization.ipynb`, Section 6)

One-at-a-time (OAT) analysis across battery parameters (threshold strategy for speed):

- **Capacity (S_max 5–15 kWh):** savings grow with capacity but plateau around 10–12 kWh for a 3 500 kWh/yr household.
- **Min SOC (S_min 0–20%):** savings decrease roughly linearly; 10% is a reasonable trade-off between battery longevity and arbitrage.
- **Max power (P_max 2–10 kW):** LP shows monotone improvement; threshold strategy degrades at high P_max (greedy over-charging).
- **2D grid S_max × P_max:** identifies the bottleneck region where increasing one parameter without the other yields no benefit.

### 6. Payback analysis (`03_optimization.ipynb`, Section 7)

Assumptions: battery cost 4 000 EUR installed (BYD LFP 10 kWh), 10-year calendar warranty, 6 000 EFC cycle warranty, 3%/yr electricity price growth, 2.5%/yr capacity degradation.

| Scenario | Simple payback |
|---|---|
| Full period average (2022–2025) | ~27 years |
| Normal market only (2023–2025) | ~39 years |
| Any tested consumption (2 000–9 000 kWh/yr) | > 10 years |

The battery is **lightly cycled** (~200 EFC/year vs 6 000 EFC warranty) — the binding constraint is the 10-year calendar warranty, not physical wear.

**Section 7b** separates the crisis year from the normal market.  
**Section 7c** computes EFC-based lifespan vs calendar warranty.  
**Section 7d** tests consumption sensitivity from a small flat (2 000 kWh) to a household with an EV (9 000 kWh).

### 7. Main conclusion

> **Pure EPEX price arbitrage does not pay off a residential battery in Belgium under normal market conditions.** The simple payback period (27–39 years) exceeds the calendar warranty (10 years) at all tested household sizes.

The main economic driver for residential batteries in Belgium is **solar self-consumption**, not arbitrage:

| Use case | Estimated payback |
|---|---|
| Dynamic tariff, no solar *(this model)* | > 27 years |
| Dynamic tariff + flexible EV charging | ~15–20 years |
| Solar PV + battery (self-consumption) | ~5–8 years |
| Solar PV + battery + capaciteitstarief | ~4–6 years |

Since the abolition of net metering for new PV installations in Belgium (2024), storing excess solar production has become the primary use case for home batteries.

### 8. Interactive demo (`streamlit_app.py`)

Two-tab Streamlit dashboard:

**Tab 1 — Scenario Demo**
- 5 pre-curated 3-day windows (crisis peak Aug 2022, solar surplus Aug 2024, balanced spring 2025, winter winds Dec 2023, normal winter Jan 2024)
- Side-by-side comparison: day-by-day LP (no forecast) vs 72-hour LP (perfect forecast)
- Quantifies the *value of forecast*: how much more the optimizer earns when it sees across day boundaries
- Per-day breakdown table + full hourly schedule

**Tab 2 — Live Planner**
- User selects any date in 2025; window starts at 13:00 (EPEX publication time)
- First 35 hours = known EPEX prices (today 13:00–24:00 + all of tomorrow)
- Remaining 37 hours = forecast zone (shown with hatching)
- Single 72-hour LP run; price chart, charge/discharge, and SOC panels in one aligned figure

Sidebar controls: battery capacity, max power, min SOC, initial SOC, round-trip efficiency, degradation cost, annual household consumption.

---

## Data sources

| Source | Content | Period | File |
|---|---|---|---|
| ENTSO-E | Hourly BE EPEX day-ahead prices (EUR/MWh) | 2022–2025 | `prices_be.csv` |
| Open-Meteo | Hourly weather in Antwerp (temp, wind, solar, clouds) | 2022–2025 | `weather_antwerp.csv` |
| Synthetic H0 | Household load profile (relative units, scalable) | — | `load_profile.csv` |

---

## In progress

**Price forecasting module** (colleague Mathijs): a model that predicts hourly EPEX prices before the 13:00 official publication, enabling the Live Planner to use real predictions instead of historical prices as a proxy. Candidate models: LightGBM or LSTM with weather, time, and lag features.

---

## Tech stack

Python 3.13 · pandas · numpy · PuLP (CBC solver) · matplotlib · Streamlit · Jupyter Notebook
