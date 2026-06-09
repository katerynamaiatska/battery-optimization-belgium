import json

NB_PATH = r"personal_optimization/notebooks/05_solar_forecast.ipynb"

def code(source, cell_id):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }

def md(source, cell_id):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }

cells = []

# ── §0 Intro ──────────────────────────────────────────────────────────────────
cells.append(md("""\
# 05 — Solar Production Forecast

## Why forecast solar?

In the LP optimizer, solar production enters the grid balance every 15 minutes:

```
grid_import - grid_export = load + battery_charge - battery_discharge - solar
```

For a **day-ahead LP** schedule (planned at 13:00 for the next day), we need to know
**how much solar the panels will produce tomorrow**.

If we overestimate solar → battery stays too empty at sunrise, evening has no stored energy.
If we underestimate solar → battery is full at noon, solar surplus is wasted (exported at 0 EUR/kWh).

### Why solar is easier to forecast than consumption

Solar follows **physics**: the sun rises at a predictable angle, clouds block some fraction of it.
The key input is **shortwave radiation** (W/m²) — how much solar energy reaches the ground at a given hour.
A weather API (e.g. Open-Meteo) provides this as a day-ahead forecast.

Unlike consumption, solar does **not** depend on what happened yesterday.
There are no lag features needed — just radiation + time of day + season.

### What this notebook does

1. Fetch historical weather data with radiation variables
2. Explore the relationship between radiation and solar production
3. Train a model to predict `sl_productie_kwh` per 15-min slot
4. Evaluate accuracy (day-time only — nighttime zeros are trivial)
5. Export predictions for use in `03_optimization_solar.ipynb`
""", "sf-intro"))

# ── §1 Imports ────────────────────────────────────────────────────────────────
cells.append(md("## §1. Imports & paths", "sf1-md"))
cells.append(code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import json
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Paths
DATA_FILE    = Path("../../Data/real_load/Final/optimization_input.csv")
WEATHER_FILE = Path("../../Data/real_load/Final/weather_solar.csv")   # new file with radiation
OUTPUT_FILE  = Path("../../Data/real_load/Final/solar_forecast.csv")

print("Imports OK")
""", "sf1-code"))

# ── §2 Fetch weather with radiation ──────────────────────────────────────────
cells.append(md("""\
## §2. Fetch weather data with radiation

The existing `weather_personal.csv` has only temperature and cloudcover.
For solar forecasting we need **shortwave_radiation** — the total solar energy
reaching the ground surface (W/m²). This is the single best predictor.

We fetch it from Open-Meteo (same free API as before) and cache to `weather_solar.csv`.
""", "sf2-md"))
cells.append(code("""\
if WEATHER_FILE.exists():
    weather = pd.read_csv(WEATHER_FILE, index_col=0, parse_dates=True)
    print(f"Loaded from cache: {len(weather)} rows, columns: {weather.columns.tolist()}")
else:
    print("Fetching from Open-Meteo...")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=51.2194&longitude=4.4025"          # Antwerp
        "&start_date=2024-11-01&end_date=2026-04-06"
        "&hourly=temperature_2m,cloudcover,shortwave_radiation,direct_radiation"
        "&timezone=Europe/Brussels"
    )
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())

    hourly = data["hourly"]
    weather = pd.DataFrame({
        "temperature_2m":    hourly["temperature_2m"],
        "cloudcover":        hourly["cloudcover"],
        "shortwave_radiation": hourly["shortwave_radiation"],   # W/m² — main feature
        "direct_radiation":  hourly["direct_radiation"],        # direct sunlight component
    }, index=pd.to_datetime(hourly["time"]))
    weather.index.name = "time"
    weather.to_csv(WEATHER_FILE)
    print(f"Saved: {len(weather)} rows -> {WEATHER_FILE}")

weather.head(3)
""", "sf2-code"))

# ── §3 Load & merge ───────────────────────────────────────────────────────────
cells.append(md("""\
## §3. Load solar data & merge with weather

`sl_productie_kwh` is solar production per 15-min slot (kWh).
Weather is hourly → we resample to 15-min with forward-fill.
""", "sf3-md"))
cells.append(code("""\
# Load 15-min household data
df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
solar = df[["sl_productie_kwh"]].copy()
print(f"Solar data: {len(solar)} rows, {solar.index.min().date()} to {solar.index.max().date()}")
print(f"Missing values: {solar['sl_productie_kwh'].isna().sum()}")

# Resample weather hourly -> 15-min (repeat each hour 4 times)
w15 = weather.resample("15min").ffill()
w15 = w15.ffill().bfill()   # fill any edge NaNs

# Merge
merged = solar.join(w15, how="left")
merged = merged.dropna()
print(f"After merge: {len(merged)} rows, {merged.isna().sum().sum()} NaNs")
merged.head(3)
""", "sf3-code"))

# ── §4 Explore ────────────────────────────────────────────────────────────────
cells.append(md("""\
## §4. Explore: radiation vs solar production

Before training, let's see how well radiation already predicts solar output.
We expect a near-linear relationship during daytime.

**Key question:** is `shortwave_radiation` enough, or do we need more features?
""", "sf4-md"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Scatter: radiation vs solar production (daytime only)
daytime = merged[merged["shortwave_radiation"] > 10]
axes[0].scatter(daytime["shortwave_radiation"], daytime["sl_productie_kwh"],
                alpha=0.15, s=5, color="#e67e22")
axes[0].set_xlabel("Shortwave radiation (W/m²)")
axes[0].set_ylabel("Solar production (kWh / 15 min)")
axes[0].set_title("Radiation vs solar output — daytime slots")

# Monthly average solar production
merged["month"] = merged.index.month
monthly = merged.groupby("month")["sl_productie_kwh"].mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
axes[1].bar(monthly.index, monthly.values, color="#f39c12")
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels([month_names[m-1] for m in range(1, 13)])
axes[1].set_ylabel("Avg solar production (kWh / 15 min)")
axes[1].set_title("Monthly average solar output")

plt.tight_layout()
plt.show()

print(f"Daytime slots (radiation > 10 W/m²): {len(daytime)} of {len(merged)} total")
print(f"Correlation radiation <-> solar: {daytime['shortwave_radiation'].corr(daytime['sl_productie_kwh']):.3f}")
""", "sf4-code"))

# ── §5 Features ───────────────────────────────────────────────────────────────
cells.append(md("""\
## §5. Feature engineering

For solar we need:
- **Radiation features**: `shortwave_radiation`, `direct_radiation` — the physical driver
- **Time features**: hour of day, month, day of year — solar angle changes with season/time
- **Cloud cover**: extra signal on top of radiation (radiation already encodes clouds partially)

No lag features needed — solar depends on weather today, not yesterday.
""", "sf5-md"))
cells.append(code("""\
# Time features
merged["hour"]       = merged.index.hour
merged["minute"]     = merged.index.minute
merged["slot"]       = merged["hour"] * 4 + merged["minute"] // 15   # 0..95
merged["month"]      = merged.index.month
merged["day_of_year"] = merged.index.day_of_year

# Sin/cos encoding so model knows that slot 95 is close to slot 0 (midnight)
merged["slot_sin"]  = np.sin(2 * np.pi * merged["slot"] / 96)
merged["slot_cos"]  = np.cos(2 * np.pi * merged["slot"] / 96)
merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)

FEATURES = [
    "shortwave_radiation",   # main driver
    "direct_radiation",      # direct vs diffuse split
    "cloudcover",            # extra cloud signal
    "temperature_2m",        # panel efficiency (minor effect)
    "slot_sin", "slot_cos",  # time of day
    "month_sin", "month_cos", # season
    "day_of_year",           # solar angle changes across the year
]

TARGET = "sl_productie_kwh"

print(f"Features: {FEATURES}")
print(f"Dataset: {len(merged)} rows")
""", "sf5-code"))

# ── §6 Train/test split ───────────────────────────────────────────────────────
cells.append(md("""\
## §6. Train / test split

Same approach as the consumption forecast:
- **Train**: Nov 2024 – Dec 2025 (first ~14 months)
- **Test**: Jan 2026 – Apr 2026 (last ~3.5 months)

We never mix future data into training — this mimics how a real forecast would work.

No 7-day warmup needed here (no lag features), so we keep all rows.
""", "sf6-md"))
cells.append(code("""\
SPLIT_DATE = "2026-01-01"

train = merged[merged.index < SPLIT_DATE]
test  = merged[merged.index >= SPLIT_DATE]

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"Train: {len(train)} rows  ({train.index.min().date()} to {train.index.max().date()})")
print(f"Test:  {len(test)} rows   ({test.index.min().date()} to {test.index.max().date()})")
print(f"Solar production > 0 in test: {(y_test > 0).sum()} slots ({(y_test > 0).mean()*100:.1f}%)")
""", "sf6-code"))

# ── §7 Train model ────────────────────────────────────────────────────────────
cells.append(md("""\
## §7. Train model

Same model as for consumption: **HistGradientBoostingRegressor**.

For solar, it should be even easier to fit because the relationship between
radiation and output is near-linear and the noise is lower.

We clip predictions to 0 — solar panels cannot produce negative energy.
""", "sf7-md"))
cells.append(code("""\
model = HistGradientBoostingRegressor(
    max_iter=200,
    max_leaf_nodes=31,
    learning_rate=0.05,
    random_state=42,
)
model.fit(X_train, y_train)

# Predict and clip to 0 (no negative solar)
y_pred_train = np.maximum(model.predict(X_train), 0)
y_pred_test  = np.maximum(model.predict(X_test),  0)

# Evaluate on ALL slots
mae_all  = mean_absolute_error(y_test, y_pred_test)
rmse_all = mean_squared_error(y_test, y_pred_test) ** 0.5

# Evaluate on DAYTIME only (radiation > 10 W/m²)
# Nighttime zeros are trivially easy to predict — they inflate accuracy
day_mask = test["shortwave_radiation"] > 10
mae_day  = mean_absolute_error(y_test[day_mask], y_pred_test[day_mask])
rmse_day = mean_squared_error(y_test[day_mask], y_pred_test[day_mask]) ** 0.5
baseline_day = mean_absolute_error(y_test[day_mask], [y_test[day_mask].mean()] * day_mask.sum())

print("── All slots ──────────────────────────────")
print(f"  MAE  = {mae_all:.4f} kWh/slot")
print(f"  RMSE = {rmse_all:.4f} kWh/slot")
print()
print("── Daytime only (radiation > 10 W/m²) ────")
print(f"  MAE  = {mae_day:.4f} kWh/slot")
print(f"  RMSE = {rmse_day:.4f} kWh/slot")
print(f"  Naive baseline MAE (predict mean): {baseline_day:.4f} kWh/slot")
print(f"  MAE improvement vs naive: {(1 - mae_day/baseline_day)*100:.1f}%")
""", "sf7-code"))

# ── §8 Visual check ───────────────────────────────────────────────────────────
cells.append(md("""\
## §8. Visual check — predicted vs actual

Let's plot a few days to see how well the forecast tracks reality.
""", "sf8-md"))
cells.append(code("""\
# Pick one sunny and one cloudy week in test period
test["solar_pred"] = y_pred_test

fig, axes = plt.subplots(2, 1, figsize=(14, 7))
for ax, start in zip(axes, ["2026-01-20", "2026-03-10"]):
    window = test[start : (pd.Timestamp(start) + pd.Timedelta(days=4)).strftime("%Y-%m-%d")]
    ax.plot(window.index, window["sl_productie_kwh"], label="Actual",    color="#e67e22", lw=1.5)
    ax.plot(window.index, window["solar_pred"],       label="Forecast",  color="#2980b9", lw=1.5, ls="--")
    ax.set_title(f"Week starting {start}")
    ax.set_ylabel("kWh / 15 min")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
""", "sf8-code"))

# ── §9 Feature importance ─────────────────────────────────────────────────────
cells.append(md("""\
## §9. Feature importance

Which features matter most? We use permutation importance (same as in `04_consumption_forecast.ipynb`):
shuffle one feature at a time and measure how much error increases.
""", "sf9-md"))
cells.append(code("""\
perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    "feature":    FEATURES,
    "importance": perm.importances_mean,
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance_df["feature"], importance_df["importance"], color="#2980b9")
ax.set_xlabel("Mean increase in MAE when feature is shuffled")
ax.set_title("Feature importance — solar forecast")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()
""", "sf9-code"))

# ── §10 Export ────────────────────────────────────────────────────────────────
cells.append(md("""\
## §10. Export predictions

Save predictions for all rows (train + test) to `solar_forecast.csv`.
This will be loaded in `03_optimization_solar.ipynb` for the LP backtest.
""", "sf10-md"))
cells.append(code("""\
# Predict for the full dataset
all_preds = np.maximum(model.predict(merged[FEATURES]), 0)
merged["solar_pred"] = all_preds

output = merged[["sl_productie_kwh", "solar_pred"]].copy()
output.to_csv(OUTPUT_FILE)
print(f"Saved: {len(output)} rows -> {OUTPUT_FILE}")
print(output.describe().round(4))
""", "sf10-code"))

# ── Assemble notebook ─────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Created: {NB_PATH} ({len(cells)} cells)")
