# Home Battery Optimization under Dynamic Electricity Tariffs
**Data Science Internship — Belgium | 2026**
*Kateryna Maiatska*

---

## Project idea

Belgian consumers on a dynamic electricity contract pay a price that changes every hour, based on EPEX SPOT day-ahead market prices. Most of them don't know when to charge or discharge their home battery to actually save money.

This project builds an optimization system that computes the ideal charge/discharge schedule for a home battery over 24 hours, using real historical prices. The result is a quantified saving in euros for a typical Belgian household.

A comparable tool exists for the Netherlands ([energie.theoxygent.nl](https://energie.theoxygent.nl)) — Does a comparable public solution exist for Belgium?

---

## Approach

The optimization is formulated as a **Linear Programming (LP)** problem using the PuLP library:

**Minimize** `Σ p(t) × (load(t) + charge(t) − discharge(t))` for t = 1..24

Subject to:
- Battery state of charge stays within capacity limits
- Charge/discharge power stays within hardware limits
- Energy balance holds for every hour

Three scenarios are compared to quantify value:

| Scenario                  | Description                          |
|---------------------------|--------------------------------------|
| Fixed tariff              | Average price, no battery — baseline |
| Dynamic, no battery       | Dynamic tariff, no optimization      |
| Dynamic + battery (LP)    | Optimal charge/discharge schedule    |

The difference in euros per month and per year is the measurable output.

---

## Data sources

Source:     ENTSO-E,  Open-Meteo Historical API,   Household consumption data  
Content:    Hourly BE day-ahead prices (EUR/MWh),  Hourly weather in Antwerp (temperature, wind, solar radiation, cloud cover),  Real hourly consumption profile  
Period: 2022–2025, 2022–2025, TBD  
File: prices_be.csv, weather_antwerp.csv, Not yet available  

Battery parameters are user-defined: capacity (kWh), max power (kW), efficiency (%).

---

## Expected output

Simple website for users with price forecasting statistics and personal recommendations.

- **One concrete number** statistic: example, "A Belgian household with a 10 kWh battery saves ~X euros/year on a dynamic tariff"
- **Hourly chart** policy: price curve + charge/discharge schedule — visual proof the system works
- **Reproducible Jupyter Notebook** pipeline, ready for extension or integration

---

## Tech stack

Python 3.10+ · pandas · numpy · PuLP · matplotlib · plotly · Jupyter Notebook

---

## Future extension

The **price forecasting module**: a model that predicts hourly prices before the official 13:00 publication, making the system proactive rather than reactive. Suitable models: LightGBM or LSTM with weather, time, and lag features. (I will use the result from my colleague)
