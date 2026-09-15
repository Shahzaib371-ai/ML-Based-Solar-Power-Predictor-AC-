# Can We Predict Solar Power Output With Just a Straight Line? A Linear Regression Case Study

*How I built, tested, and compared six regression models to forecast AC power from a real solar plant — using nothing but on-site sensors, public weather data, and gradient descent written from scratch.*

## The Question

Solar plant operators need to know, hours in advance, how much power a plant will actually feed into the grid. Too little forecasting accuracy and utilities either waste backup generation capacity or risk blackouts. My Final Year Project set out to answer a narrow but practical version of that problem:

> **Can a simple linear model predict a solar plant's hourly AC power output accurately enough to be useful — and does it matter whether the input weather data comes from the plant's own sensors or from a free public weather API?**

To answer this, I trained the *same* linear regression model three different ways (the Normal Equation, Batch Gradient Descent, and Stochastic Gradient Descent) on *two* different feature sets — one built from the plant's physical sensors, one built from Open-Meteo's public historical weather archive — and compared how each combination performed on a held-out test week.

## The Data

The raw data comes from a real solar power plant (Plant 1) with two files: a **generation log** recording DC and AC power output every 15 minutes from multiple inverters, and a **weather sensor log** recording ambient temperature, module temperature, and solar irradiation over the same period (15 May – 17 June 2020). To bring in an independent, free data source, I also pulled historical shortwave radiation, 2-metre temperature, and cloud cover for the plant's approximate coordinates (14.82°N, 78.28°E) from the **Open-Meteo Historical Weather API**.

**Table 1 — Data Preparation**

| Step | Value |
|---|---|
| Raw generation rows (Plant 1) | 68,778 |
| Raw sensor rows (Plant 1) | 3,182 |
| Timestamps present in only one file | 26 (1 generation-only, 25 sensor-only) |
| Hourly rows after resampling | 816 |
| Hourly rows with missing values (pre-cleaning) | 20 |
| Open-Meteo rows downloaded (merged, matching hours) | 816 |

The inverter-level generation data was first summed across inverters, merged with the sensor data on timestamp, and resampled from 15-minute to **hourly** resolution to match the resolution available from Open-Meteo. The 20 hourly rows left with gaps were filled using **linear interpolation** — a reasonable choice here because both power output and weather variables change smoothly over the course of a day, so estimating a missing hour from its neighbours introduces very little distortion.

## The Location Check

Before trusting an external weather source as a stand-in for on-site sensors, I needed to confirm the two data sources actually agree on *when* things happen — not just *how much*. So I plotted sensor-measured irradiation against Open-Meteo's shortwave radiation for three consecutive days and compared the hour at which each source recorded peak solar output.

| Date | Sensor Peak Hour | Open-Meteo Peak Hour | Difference |
|---|---|---|---|
| 15 May 2020 | 12:00 | 12:00 | 0 hours |
| 16 May 2020 | 12:00 | 12:00 | 0 hours |
| 17 May 2020 | 11:00 | 12:00 | 1 hour |

The two sources lined up almost exactly, with peak solar output landing at solar noon and drifting by no more than one hour across the sample. That was the green light to treat Open-Meteo as a legitimate, free substitute for on-site sensors — which is exactly what "Set B" tests below.

![Exploratory analysis: irradiation, temperature, DC/AC conversion, and daily power trend](project_overview.png)
*Figure 1 — Exploratory relationships in the cleaned hourly dataset: AC power tracks irradiation almost linearly, module temperature rises with ambient temperature, DC-to-AC conversion is near-perfectly linear (confirming stable inverter efficiency), and generation peaks around solar noon.*

## Building the Models

I defined two feature sets, both scaled to zero mean and unit variance and both including cyclical hour-of-day encodings (`sin_hour`, `cos_hour`) so the model can learn the daily generation cycle:

- **Set A (on-site sensors):** irradiation, module temperature, ambient temperature, sin(hour), cos(hour)
- **Set B (public weather API):** shortwave radiation, 2 m air temperature, cloud cover, sin(hour), cos(hour)

Each feature set was fit three ways: the closed-form **Normal Equation**, **Batch Gradient Descent** (α = 1e-5, 3,000 iterations), and **Stochastic Gradient Descent** (α = 1e-3, 100 epochs). Training used 27 days of data (15 May – 10 June); the final week (11–17 June) was held out purely for testing.

## The Results

**Table 2 — Test-Set RMSE (kW)**

| Solver | Features | All Hours | Daytime Only |
|---|---|---|---|
| Normal Equation | Set A | **561.38** | 733.21 |
| Batch GD (α = 1e-5, 3,000 iters) | Set A | 777.10 | 1,010.16 |
| Stochastic GD (α = 1e-3, 100 epochs) | Set A | 689.51 | 894.45 |
| Normal Equation | Set B | 2,636.49 | 3,424.88 |
| Batch GD (α = 1e-5, 3,000 iters) | Set B | 2,615.81 | 3,400.68 |
| Stochastic GD (α = 1e-3, 100 epochs) | Set B | **2,590.22** | 3,358.96 |

![Test-set RMSE comparison across all six solver and feature-set combinations](rmse_comparison.png)
*Figure 2 — Set A (built from the plant's own sensors) consistently produces roughly 4–5× lower test error than Set B (built entirely from free public weather data), regardless of which solver trains it.*

**Table 3 — Learned θ, Set A**

| Coefficient | Normal Eq. | Batch GD | SGD |
|---|---|---|---|
| θ₀ (intercept) | 6796.26 | 6796.26 | 6794.31 |
| θ₁ irradiation | 8195.27 | 5536.71 | 6261.89 |
| θ₂ module temp | 8.93 | 3255.31 | 2605.87 |
| θ₃ ambient temp | -28.58 | -818.50 | -840.67 |
| θ₄ sin(hour) | -30.23 | 69.11 | -49.56 |
| θ₅ cos(hour) | -369.44 | -405.97 | -361.73 |
| **Max \|θ_GD − θ_normal\|** | — | **3246.38** | — |

Two things stand out immediately. First, **daytime-only RMSE is always worse than all-hours RMSE** — that's expected, since the "all hours" number is inflated with easy-to-predict nighttime zeros, while the daytime number reflects error only when the model is actually doing meaningful forecasting work. Second, **the intercept (θ₀) is essentially identical across all three solvers**, but the irradiation and temperature weights diverge substantially between the Normal Equation and the two gradient-descent methods — a difference of over 3,200 units at its largest. That gap is a direct symptom of gradient descent not having fully converged within the iteration/epoch budget used here; the two features (irradiation and module temperature) are correlated with each other, which flattens the cost surface along certain directions and slows convergence exactly where you'd need more iterations to close the gap with the exact solution.

## Task 5.2: Which Model Would I Actually Deploy?

Given the choice between six trained models, I'd deploy the **Normal Equation fit on Set A**. Three reasons:

1. **Lowest error, full stop.** It posts the best RMSE on both the all-hours (561 kW) and daytime-only (733 kW) test splits — no other combination gets close.
2. **No convergence risk.** The Normal Equation solves for θ exactly in one step; there's no learning rate to tune and no risk of stopping before convergence, unlike Batch GD, which was still 3,246 units away from the exact answer even after 3,000 iterations.
3. **It only costs three sensors.** Set A needs irradiation, module temperature, and ambient temperature — all of which a solar plant already has installed for monitoring. There's no operational reason to trade that accuracy away for Open-Meteo's free-but-noisier Set B data.

That said, Set B isn't worthless — it's the fallback for sites *without* on-site sensors, or for forecasting hours or days ahead when live sensor readings for the future obviously don't exist yet and a weather forecast API is the only option.

## Wrapping Up

This project confirmed something reassuring for a beginner in both AI and power systems: you don't need a deep neural network to get useful solar forecasts. A five-feature linear regression, fit with a one-line matrix equation, cuts test error to roughly 561 kW on a plant generating up to ~25 MW at peak — good enough to be genuinely useful for grid planning. The next steps on my list are folding in cloud-cover forecasts for multi-hour-ahead prediction, testing whether a small neural network meaningfully beats this linear baseline, and adding anomaly detection to catch inverter faults automatically.

---
*All code, saved model weights (`theta_B.npy`), and feature scalers (`scaler_B_mean.npy`, `scaler_B_std.npy`) are available in the project repository.*
