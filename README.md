# Home Battery Optimization under Dynamic Electricity Tariffs
**Data Science Internship — Belgium | 2026**

---

## Project idea

Belgian consumers on a dynamic electricity contract pay a price that changes every hour, based on EPEX SPOT day-ahead market prices. Most of them don't know when to charge or discharge their home battery to actually save money.

This project builds an optimization system that computes the ideal charge/discharge schedule for a home battery, uses real historical prices to quantify savings over 4 years, and provides an interactive demo for households considering a battery purchase.

A comparable tool exists for the Netherlands ([energie.theoxygent.nl](https://energie.theoxygent.nl)) — this project asks the same question for Belgium.

---

## Part 1 — Synthetic household data, no solar panels (`optimization/`)

The first part builds and evaluates an LP battery optimization system using **synthetic household load profiles and public EPEX day-ahead price data** for Belgium (2022–2025). No solar panels — pure price arbitrage on a standard H0 consumption profile (3 500 kWh/yr baseline).

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

## Part 2 — Real household data with solar panels (`personal_optimization/`)

The second part of the project applies the LP model to **measured data from a real Belgian household**
(SOFAR ME3000SP inverter · BYD LFP 5 kWh / 3 kW · solar PV · dynamic EPEX contract).
The dataset spans **Nov 2024 – Apr 2026 (522 days, 15-min resolution)** and includes grid import/export,
solar production, battery charge/discharge, and SOFAR SOC logs.

The main practical goal of Part 2: to test on real data whether it is worthwhile for this specific household to **switch from the day/night tariff to dynamic EPEX**, and what role the battery plays under different control scenarios.

> EV charging (~2 400 kWh/yr) is excluded from all saving figures — it does not interact with the battery system and cancels out identically across all scenarios.

---

### Data preparation (`personal_optimization/notebooks/01_eda_real_load.ipynb`)

- Solar production reconstructed from energy balance (CT clamp data — direct self-consumption is invisible)
- SOFAR inverter freeze events identified: **89 stuck periods on 56 days/year** — battery delivers zero energy
- `bat_laden` glitches clipped to physical limits (0–0.75 kWh/slot)
- EPEX day-ahead prices joined at 15-min resolution (Belgian local time)
- Exported to `optimization_input.csv` — single clean file used by all downstream steps

---

### LP optimization with solar (`personal_optimization/notebooks/03_optimization_solar.ipynb`)

The LP objective is extended for solar self-consumption:

```
Minimize  Σ [ p(t) × g_in(t) − p_inj(t) × g_out(t) + deg_cost × c(t) ]
```

Solar surplus charges the battery for free; injection price = 0 EUR/kWh (no net-metering compensation since 2024).

**Backtest results — 522 days, BYD 5 kWh / 3 kW, DEG = 0.10 EUR/kWh, markup = 0.17 EUR/kWh**

| Scenario | Annual electricity cost | Battery saving vs solar-only baseline |
|---|---|---|
| Solar only, no battery — dag/nacht tariff | reference | — |
| **LP dag/nacht** (optimal, current contract) | reference − 92 EUR | **+92 EUR/yr** |
| **Real SOFAR today** (actual inverter, dag/nacht) | reference − 4 EUR | **+4 EUR/yr** |
| **LP EPEX + markup** (optimal, dynamic tariff) | reference − 120 EUR | **+120 EUR/yr** |

**Why the real SOFAR barely contributes (+4 EUR/yr):**
- Freeze events on 56 days/year eliminate all saving on those days
- Built-in rule charges from the grid at sub-optimal times
- LP optimal control on the same hardware would save +92 EUR/yr

**Why LP EPEX saving looks larger but isn't necessarily better:**

Each saving figure above is measured against its **own tariff baseline** (EPEX no-battery vs EPEX with battery; dag/nacht no-battery vs dag/nacht with battery).
To compare strategies fairly, a common baseline is needed:

| Question | Metric | Answer |
|---|---|---|
| Is adding a battery to an EPEX contract worth it? | LP EPEX vs EPEX no-battery | **+120 EUR/yr** |
| Should I switch from dag/nacht + no battery to EPEX + LP battery? | LP EPEX vs dag/nacht no-battery | **+47 EUR/yr net** |
| Does switching to EPEX alone (no battery) save money? | EPEX no-battery vs dag/nacht no-battery | **−73 EUR/yr** (costs more) |
| At what markup does LP EPEX beat LP dag/nacht? | breakeven markup | **≤ 0.162 EUR/kWh** (§6) |

**Saving decomposition — LP EPEX vs EPEX no-battery baseline (§6c):**

| Component | EUR/yr | Note |
|---|---|---|
| Solar self-consumption saving | ~195 EUR | 99% of gross — battery stores free solar surplus |
| Price arbitrage saving | ~1 EUR | 1% — only 4% of days have spread > threshold |
| **Gross saving** | **~196 EUR** | before battery wear |
| Battery wear cost (DEG) | −76 EUR | 39% of gross — dominant cost factor |
| **Net saving ★** | **~120 EUR** | LP EPEX vs EPEX no-battery |

> The net saving vs dag/nacht no-battery (different baseline — the "should I switch?" question) is **~47 EUR/yr**,
> computed as: 120 EUR (battery on EPEX) − 73 EUR (EPEX tariff penalty vs dag/nacht).

In 2025, **541 hours** had negative EPEX prices (min −0.46 EUR/kWh) — the LP charges automatically at negative prices; the SOFAR controller does not.

---

### Interactive Streamlit app (`personal_optimization/app.py`)

Six-tab dashboard running on real household data:

| Tab | Content |
|---|---|
| **EDA — Real data** | Monthly energy flows, price distributions, average day profiles, SOC by hour |
| **Validation 2026** | Real SOFAR vs LP dag/nacht vs LP EPEX — day-by-day comparison on selected dates |
| **Backtest** | Full 522-day backtest, monthly saving chart, markup sensitivity |
| **Forecast** | LP schedule for any selected 3-day window with real EPEX prices |
| **Battery Calculator** | Custom battery parameters (capacity, power, cost, EFC, markup) → annual saving, payback estimate |
| **ML Forecast LP** | 35-hour LP schedule using ML consumption and solar forecasts; both dag/nacht and EPEX shown side by side; battery parameters and markup set by sliders; KPIs show difference vs SOFAR |

**Battery Calculator** computes DEG cost from user inputs (`battery_cost / (EFC × S_MAX)`),
runs a full LP backtest on all historical data, and outputs net saving with payback period
(marked as *ideal scenario*: perfect day-ahead price foresight, constant battery capacity).

---

## Data sources

**Part 1 — synthetic/public data**

| Source | Content | Period | File |
|---|---|---|---|
| ENTSO-E | Hourly BE EPEX day-ahead prices (EUR/MWh) | 2022–2025 | `prices_be.csv` |
| Open-Meteo | Hourly weather in Antwerp (temp, wind, solar, clouds) | 2022–2025 | `weather_antwerp.csv` |
| Synthetic H0 | Household load profile (relative units, scalable) | — | `load_profile.csv` |

**Part 2 — real household measurements**

| Source | Content | Period | File |
|---|---|---|---|
| Fluvius P1 port | 15-min grid import/export, day/night tariff flag | Nov 2024 – Apr 2026 | `optimization_input.csv` |
| SOFAR ME3000SP | Battery charge/discharge, SOC (logged days only) | Nov 2024 – Apr 2026 | `overall_verrijkt.csv` |
| ENTSO-E (15-min) | EPEX day-ahead prices, Belgian local time | Nov 2024 – Apr 2026 | joined in `optimization_input.csv` |

---

## Do we need forecasting?

A key finding of Part 2 is that the value of forecasting is much smaller than it appears.

### EPEX price — no forecasting needed

Day-ahead prices are published at ~13:00 for the full next day (96 quarter-hour slots).
The LP already uses exactly these prices — there is no forecast error on price.
This is not a simplification: it is the realistic operating condition for any real controller.

### Consumption — matters for day-ahead LP

Battery dispatch decisions are primarily driven by price, but consumption enters the grid balance
constraint and affects LP decisions when the actual load differs from the forecast used at schedule time.

**Empirical test (§10, `03_optimization_solar.ipynb`):**
A rough average-profile forecast (mean per weekday × 15-min slot) was used as LP input;
the resulting schedule was evaluated at actual load over the full 522-day dataset.

| | EUR/yr |
|---|---|
| LP with actual load (lower bound) | reference |
| LP with average-profile forecast (eval at actual) | +85 EUR/yr (+5.85%) |

The 85 EUR/yr overhead from load forecast error is enough to **erase the 47 EUR/yr net saving**
vs a dag/nacht contract without battery.

**Why the gap exists:** 99% of saving comes from solar self-consumption (store solar surplus at noon,
use in evening). When the LP is given a wrong load profile, it may discharge at the wrong time —
and any surplus exported to the grid earns 0 EUR/kWh (PRICE_INJ = 0), so the battery wears
for no benefit.

**Two solutions:**
- **MPC** (re-optimise every 15 min using real-time P1 consumption data) — eliminates forecast error
  entirely, but requires a continuously running controller.
- **Better ML forecast** (lag features, day-of-week patterns) — reduces the gap; becomes worthwhile
  when annual savings exceed ~200 EUR/yr.

### Solar production — an ML forecast adds overhead, not value

The only decision that genuinely requires solar information is:
> *How full should the battery be at sunrise — should I leave room for solar surplus?*

For this, a rough weather signal (sunny / cloudy) from a free API is sufficient.

**Empirical result (§11, `03_optimization_solar.ipynb`):** A full ML solar forecast model
(HistGradientBoostingRegressor on irradiance + time features, ~26% improvement over naive baseline)
was added on top of the ML consumption forecast and evaluated at actual solar output.

| Scenario | vs current SOFAR |
|---|---|
| LP dag/nacht + ML consumption forecast | +32 EUR/yr |
| LP dag/nacht + ML consumption + ML solar | **+5 EUR/yr** |
| LP EPEX + ML consumption forecast | −17 EUR/yr |
| LP EPEX + ML consumption + ML solar | −42 EUR/yr |

Adding the ML solar forecast costs **~27 EUR/yr in additional overhead** — it does not improve results,
it makes them worse. The reason: `sl_productie_kwh` is a derived quantity (grid injection + battery
charging), not a direct measurement. SOFAR freeze events (56 days/yr) and discrete charging steps
(0 / 0.25 / 0.5 kWh/slot) add noise that the model cannot learn around; daytime correlation
with irradiance is only 0.56.

**The strongest practical conclusion from §11:** LP dag/nacht with both ML forecasts achieves
**+5 EUR/yr vs SOFAR** — essentially the same result as the current system. A complete day-ahead
LP pipeline with two trained ML models and all the associated infrastructure delivers no benefit
over the simple SOFAR threshold rule at this scale. The forecasting complexity cancels itself out.

### Multi-day horizon — marginal benefit

A multi-day LP comparison was not computed in this project (`run_lp_epex_forecast()` was
prototyped but not evaluated). The qualitative argument is structural:

- **96% of days have an EPEX spread below the profitable threshold** (§6c) — cross-day arbitrage
  has almost no room to add value beyond single-day optimisation.
- **Solar self-consumption is intra-day** (store at noon, discharge in the evening) — a 24-hour
  horizon captures it completely.
- The only cross-day decision is morning SoC, which is handled by the binary weather signal above.

*Quantifying the exact gap between 1-day and multi-day LP remains future work.*

### When does forecasting become worth the investment?

| Annual net saving | Verdict |
|---|---|
| < 100 EUR/yr (markup ≥ 0.17) | Any API or infrastructure cost exceeds the saving — not worth it |
| 150–250 EUR/yr (markup ~0.13) | A free weather signal (Open-Meteo) is sufficient; paid solar forecast unjustified |
| > 300 EUR/yr (larger battery or lower markup) | Full pipeline with solar + consumption forecast starts to make economic sense |

**Practical conclusion:** for the household studied, a viable real-world controller requires one of:
1. **MPC** — fetch EPEX prices at 13:00, re-run LP every 15 min with real-time P1 consumption data;
   binary weather signal for morning SoC decision. Zero forecast error, but needs 24/7 infrastructure.
2. **Threshold rule** — charge when price < threshold, discharge when price > threshold;
   does not depend on any forecast; simpler and more robust at this saving level. This is what
   the current SOFAR controller approximates (when it is not frozen).
3. **Day-ahead LP + ML forecasts** — the complete pipeline (ML consumption + ML solar) was tested
   in §11 and yields +5 EUR/yr vs SOFAR on the dag/nacht tariff: effectively no improvement over
   the current system. Adding an ML solar forecast on top of ML consumption makes results worse,
   not better. This option only becomes worthwhile at savings > ~200 EUR/yr.

### Price forecasting module (in progress)

A model predicting hourly EPEX prices *before* the 13:00 publication is being developed
as a separate workstream. Candidate models: LightGBM or LSTM with weather, time-of-day,
and lag features. This would benefit the **Live Planner** tab (Part 1), which currently uses
historical prices as a proxy for the forecast zone.

---

### Impact of forecasting — quantitative results (§10–§11)

To measure the real cost of forecast errors, the LP backtest was run three ways: with actual load and solar (lower bound), with ML consumption forecast, and with both ML forecasts. Results compared on a shared 513-day dataset.

**Forecast overhead (excess cost relative to ideal LP):**

| Consumption forecast | Solar forecast | Tariff | Overhead |
|---|---|---|---|
| Average profile | actual | EPEX | ~+85 EUR/yr |
| ML forecast | actual | EPEX | ~+60 EUR/yr |
| ML forecast | ML forecast | EPEX | ~+85 EUR/yr |
| Average profile | actual | dag/nacht | ~+30 EUR/yr |
| ML forecast | actual | dag/nacht | ~+57 EUR/yr |
| ML forecast | ML forecast | dag/nacht | ~+84 EUR/yr |

The ML consumption forecast reduces overhead vs average profile (~+60 instead of ~+85 EUR/yr for EPEX), but adding the solar forecast brings it back up (+27 EUR/yr of additional noise). Reason: `sl_productie_kwh` is a derived quantity with 0.56 correlation to irradiance, not a direct measurement.

**All scenarios vs current SOFAR:**

| Scenario | vs current SOFAR | Condition |
|---|---|---|
| LP + dag/nacht, ideal (MPC) | **+88 EUR/yr** ✅ | 24/7 P1 controller |
| LP + dag/nacht + ML consumption | **+32 EUR/yr** ✅ | day-ahead LP + ML |
| LP + dag/nacht + both ML | **+5 EUR/yr** ✅ | both ML forecasts |
| LP + EPEX, ideal (MPC) | **+43 EUR/yr** ✅ | tariff switch + controller |
| LP + EPEX + ML consumption | **−17 EUR/yr** ❌ | tariff switch + ML |
| LP + EPEX + both ML | **−42 EUR/yr** ❌ | tariff switch + both ML |

> ML predictions for Nov 2024 – Dec 2025 are in-sample (model trained on those same dates) — §10/§11 results for this period are slightly optimistic, by an estimated 5–15 EUR/yr.

---

### Conclusion for this household

**What can be done now:**
- Stay on the day/night tariff — at the current markup of 0.17 EUR/kWh, EPEX costs 73 EUR/yr more without a battery.
- Fix SOFAR freeze events (firmware update or inverter replacement) — this recovers up to +88 EUR/yr that the system is currently missing.
- Smart EV charging on EPEX — the biggest untapped opportunity: ~2 400 kWh/yr charged from the grid at suboptimal times. EPEX prices are known by 13:00; no ML forecast needed.

**What is not worth doing now:**
- Switching to EPEX: markup 0.17 EUR/kWh makes EPEX more expensive even with ideal LP.
- Implementing day-ahead LP with ML forecasts: forecast overhead (~60–85 EUR/yr) exceeds the potential gain.
- Paying for a solar forecast: at current savings levels it adds ~27 EUR/yr of noise, not profit.

**When to reconsider:**

| Condition | What changes |
|---|---|
| Markup ≤ 0.12 EUR/kWh | EPEX becomes competitive |
| SOFAR without freezes | Real system approaches LP ideal |
| Solar meter installed | ML solar forecast quality improves significantly |
| Consumption increases (heat pump) | Larger flows → larger absolute LP saving |

---

### Limitations and inaccuracies

**Data:**
- `sl_productie_kwh` is a derived quantity (grid injection + battery charging), not a direct measurement. Direct self-consumption is invisible; SOFAR freeze events zero out the value during periods of non-zero solar; discrete charging steps (0 / 0.25 / 0.5 kWh/slot) create horizontal bands. Net effect: correlation with irradiance is only 0.56.
- Inverter SOC is recorded on only 18 dates in 2026 — limited base for the Streamlit ML tab.
- EV (~2 400 kWh/yr) is excluded from all calculations.

**Modelling:**
- ML forecasts for Nov 2024 – Dec 2025 are in-sample — §10/§11 results for this period are slightly inflated (estimated 5–15 EUR/yr).
- LP backtest assumes perfect schedule execution with no delays and no freeze events.
- DEG = 0.10 EUR/kWh is an estimated degradation cost, not an exact figure.

---

### Model development

Parametric sensitivity analysis shows **under which conditions** the system becomes profitable. Whether those conditions materialise — more frequent negative prices, lower equipment costs, better controllers — can only be confirmed by re-running the backtest on fresh data.

**Recommended: revisit the analysis in mid-2027 and check:**
- Whether negative EPEX price frequency has changed (541 hours in 2025).
- Whether market markup has dropped to ≤ 0.12 EUR/kWh.
- Whether injection compensation has appeared (currently 0 EUR/kWh).
- Whether capaciteitstarief has been introduced and how it affects the optimal strategy.

**Technical improvements:**
1. **MPC controller** — Python or Home Assistant running LP every 15 min with real P1 data. Eliminates forecast error entirely; requires 24/7 infrastructure.
2. **Smart EV charging** — schedule charging by EPEX prices with no ML models at all.
3. **Separate solar meter** — eliminates the need to reconstruct `sl_productie`.
4. **SOFAR update or replacement** — to eliminate 89 freeze events/year.

---

## Tech stack

Python 3.13 · pandas · numpy · PuLP (CBC solver) · matplotlib · Streamlit · Jupyter Notebook · scikit-learn · joblib
