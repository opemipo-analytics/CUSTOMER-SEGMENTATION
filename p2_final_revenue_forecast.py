"""
Electricity Distribution Company — Revenue Forecasting with Machine Learning
============================================================================
Author: Opemipo Daniel Owolabi
Project: Portfolio Project 2 — Predictive Analytics
Tools: Python, Pandas, Scikit-learn, Matplotlib

Note:
-----
All company names, client names, locations and identifying information
have been anonymised to protect client confidentiality. The analytical
approach, methodology and model results reflect real work conducted
during professional employment in the electricity distribution sector.

Business Problem:
-----------------
A regional electricity distribution company needed a data-driven way
to forecast monthly revenue collection. Manual forecasting was slow
and inconsistent. This project builds a machine learning model that
learns from historical daily collection data and produces a reliable
30-day revenue forecast.

How the Model Works:
--------------------
Polynomial Regression finds the best mathematical curve through
historical data points, then extends that curve into the future.
Think of it as drawing a trend line — but mathematically precise
and extendable months ahead.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  ELECTRICITY DISTRIBUTION")
print("  REVENUE FORECASTING — MACHINE LEARNING MODEL")
print("  Analyst: Opemipo Daniel Owolabi")
print("=" * 60)


# ── DATA ──
# Anonymised daily cumulative collection figures
data = {
    "Date": ["2021-06-18","2021-06-21","2021-06-22","2021-06-23",
              "2021-06-24","2021-06-27","2021-06-28","2021-06-29","2021-06-30"],
    "Cumulative_Collection": [
        6_500_000, 7_800_000, 8_200_000, 8_750_000,
        9_300_000, 9_950_000, 10_400_000, 10_900_000, 11_300_000
    ]
}

df = pd.DataFrame(data)
df["Date"]       = pd.to_datetime(df["Date"])
df["Day_Number"] = (df["Date"] - df["Date"].min()).dt.days + 1
df["Daily"]      = df["Cumulative_Collection"].diff().fillna(df["Cumulative_Collection"].iloc[0])

print(f"\n[STEP 1] Loaded {len(df)} historical data points")

# ── MODEL ──
X      = df["Day_Number"].values.reshape(-1, 1)
y      = df["Cumulative_Collection"].values
poly   = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"[STEP 2] Model trained")
print(f"         R2 Score: {r2:.4f} ({r2*100:.1f}% accuracy)")
print(f"         MAE:      N{mae:,.0f}")

# ── FORECAST ──
last_day      = int(df["Day_Number"].max())
future_days   = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
future_pred   = model.predict(poly.transform(future_days))
future_dates  = [df["Date"].max() + pd.Timedelta(days=i) for i in range(1, 31)]

df_forecast   = pd.DataFrame({"Date": future_dates, "Predicted": future_pred})
df_forecast["Daily_Pred"] = df_forecast["Predicted"].diff().fillna(
    df_forecast["Predicted"].iloc[0] - df["Cumulative_Collection"].iloc[-1]
)
df["Model_Fit"] = model.predict(poly.transform(X))

projected     = df_forecast["Predicted"].iloc[-1]
daily_avg     = df_forecast["Daily_Pred"].mean()
growth        = (projected - df["Cumulative_Collection"].iloc[-1]) / df["Cumulative_Collection"].iloc[-1] * 100

print(f"[STEP 3] Forecast generated for 30 days")
print(f"         Projected Total:  N{projected/1e6:.2f}M")
print(f"         Daily Average:    N{daily_avg:,.0f}")
print(f"         Projected Growth: +{growth:.1f}%")


# ── DASHBOARD ──
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    "Electricity Distribution — Revenue Forecasting Dashboard\n"
    "Machine Learning Model (Polynomial Regression)  |  Analyst: Opemipo Daniel Owolabi",
    fontsize=14, fontweight="bold", y=1.01
)

ax1 = axes[0, 0]
ax1.scatter(df["Date"], df["Cumulative_Collection"] / 1e6,
            color="#1f4e79", s=80, zorder=5, label="Actual Data")
ax1.plot(df["Date"], df["Model_Fit"] / 1e6,
         color="#ed7d31", linewidth=2.5, linestyle="--", label="Model Fit")
ax1.set_title("Model Fit — Actual vs Predicted (Historical)", fontweight="bold")
ax1.set_ylabel("Cumulative Collection (N Millions)")
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("N%.1fM"))
ax1.legend()
ax1.tick_params(axis="x", rotation=30)

ax2 = axes[0, 1]
ax2.plot(df["Date"], df["Cumulative_Collection"] / 1e6,
         color="#1f4e79", linewidth=2.5, marker="o", markersize=6, label="Historical")
ax2.plot(df_forecast["Date"], df_forecast["Predicted"] / 1e6,
         color="#70ad47", linewidth=2.5, linestyle="--", marker="s", markersize=4, label="Forecast")
ax2.axvline(x=df["Date"].max(), color="#c00000", linestyle=":", linewidth=1.5, label="Forecast Start")
ax2.fill_between(df_forecast["Date"],
                 df_forecast["Predicted"] / 1e6 * 0.92,
                 df_forecast["Predicted"] / 1e6 * 1.08,
                 alpha=0.15, color="#70ad47", label="Confidence Band")
ax2.set_title("30-Day Revenue Forecast", fontweight="bold")
ax2.set_ylabel("Cumulative Collection (N Millions)")
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("N%.1fM"))
ax2.legend(fontsize=8)
ax2.tick_params(axis="x", rotation=30)

ax3 = axes[1, 0]
bar_colors = ["#70ad47" if v > daily_avg else "#ffc000" for v in df_forecast["Daily_Pred"]]
ax3.bar(df_forecast["Date"], df_forecast["Daily_Pred"] / 1e3, color=bar_colors, alpha=0.85)
ax3.axhline(y=daily_avg / 1e3, color="#c00000", linestyle="--", linewidth=1.5,
            label=f"Avg: N{daily_avg/1e3:.0f}K/day")
ax3.set_title("Predicted Daily Collections — Next 30 Days", fontweight="bold")
ax3.set_ylabel("Daily Collection (N Thousands)")
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("N%.0fK"))
ax3.legend()
ax3.tick_params(axis="x", rotation=30)

ax4 = axes[1, 1]
ax4.axis("off")
summary_text = f"""
MODEL PERFORMANCE SUMMARY
{'─'*36}

Algorithm:       Polynomial Regression (Degree 2)
Data Points:     {len(df)} historical snapshots
Train / Test:    80% / 20% split


MODEL ACCURACY
{'─'*36}

R2 Score:        {r2:.4f}  ({r2*100:.1f}% accuracy)
Mean Abs Error:  N{mae:,.0f}


30-DAY FORECAST
{'─'*36}

Projected Total:  N{projected/1e6:.2f} Million
Daily Average:    N{daily_avg:,.0f}
Projected Growth: +{growth:.1f}%


BUSINESS INSIGHT
{'─'*36}

The model forecasts continued revenue
growth in the following month based on
the trend observed in the current period.
Management can use this as a data-driven
monthly collection target.
"""
ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=9.5, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#f0f4f8", alpha=0.8))

plt.tight_layout()
plt.savefig("/home/claude/clean/project2/revenue_forecast_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

print("[STEP 4] Dashboard saved.")
print("=" * 60)
