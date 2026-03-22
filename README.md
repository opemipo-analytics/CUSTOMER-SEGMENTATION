# Electricity Distribution — Revenue Forecasting with Machine Learning

**Portfolio Project 2** — A machine learning model that forecasts future revenue collection for a regional electricity distribution company, built on real daily collection data.

> Built by **Opemipo Daniel Owolabi** — Data Analyst | Python · SQL · Power BI · Tableau  
> Faro, Portugal | opemipoowolabi001@gmail.com

---

## Note on Data

All company names, client names, locations and identifying information have been anonymised in this public version to protect client confidentiality. The analytical approach, methodology and model results reflect real work conducted during professional employment in the electricity distribution sector.

---

## Business Problem

After analysing marketer performance (Project 1), the next question management asks is:

> "Based on current trends, what will our revenue collection look like next month?"

Manual forecasting was slow and inconsistent. This project builds a machine learning model that learns from historical daily collection data and produces a reliable 30-day revenue forecast — giving management a data-driven basis for planning and target-setting.

---

## Dashboard Preview

![Revenue Forecast Dashboard](revenue_forecast_dashboard.png)

---

## How the Machine Learning Works

This project uses **Polynomial Regression** — explained simply:

| Concept | Plain English |
|---------|---------------|
| Training data | Historical daily revenue figures |
| Features (X) | Day numbers (Day 1, Day 2, Day 3...) |
| Target (y) | Cumulative revenue collected |
| Model | Finds the best mathematical curve through the data |
| Prediction | Extends that curve into the future |

Think of it like drawing a trend line — but mathematically precise and extendable months ahead.

---

## Model Results

| Metric | Value |
|--------|-------|
| Algorithm | Polynomial Regression (Degree 2) |
| R2 Score | 0.9986 — 99.9% accuracy |
| Mean Absolute Error | N58,250 |
| Train / Test Split | 80% / 20% |

---

## 30-Day Forecast Summary

| Metric | Value |
|--------|-------|
| Projected Total Revenue | N14.75 Million |
| Projected Daily Average | N114,872 per day |
| Projected Growth | +30.5% vs previous period |

---

## What the Dashboard Shows

### 1 — Model Fit: Actual vs Predicted (Historical)
Shows how well the model fits the historical data. Orange dashed line is the model curve, blue dots are real data. The closer they are, the better the model.

### 2 — 30-Day Revenue Forecast
Historical data transitions into the forecast zone. The shaded band shows a confidence range of plus or minus 8% — usable as an upper and lower bound for planning.

### 3 — Predicted Daily Collections — Next 30 Days
Bar chart of predicted daily revenue. Red dashed line marks the daily average target.

### 4 — Model Performance Summary
Clean summary panel of all key metrics — designed for easy inclusion in executive presentations.

---

## Project Structure

```
project2/
├── revenue_forecast.py              # Main ML script
├── revenue_forecast_dashboard.png   # Output: 4-panel forecast dashboard
└── README.md                        # This file
```

---

## How to Run

```bash
git clone https://github.com/opemipo-analytics/aedc-revenue-forecasting.git
cd aedc-revenue-forecasting

pip install pandas numpy matplotlib scikit-learn

python revenue_forecast.py
```

---

## Tools and Technologies

| Tool | Purpose |
|------|---------|
| Python 3 | Core scripting |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning model |
| Matplotlib | Visualisations |

---

## Skills Demonstrated

- Machine learning — training, testing and evaluating a regression model
- Feature engineering — converting dates to numeric features for ML
- Model evaluation — R2 score, Mean Absolute Error, train/test split
- Business forecasting — translating ML output into actionable revenue projections
- Data storytelling — presenting complex ML results in plain business language

---

## Other Projects

| Project | Description |
|---------|-------------|
| [Marketer Performance Analysis](https://github.com/opemipo-analytics/AEDC-MARKETERS-ANALYTICS) | Python analysis of field marketer KPIs |
| [Customer Segmentation](https://github.com/opemipo-analytics/aedc-customer-segmentation) | SQL and RFM customer segmentation |
| [Property Portfolio Analytics](https://github.com/opemipo-analytics/amcon-portfolio-analytics) | Financial property portfolio analysis |
| [Smart Meter Analytics](https://github.com/opemipo-analytics/smart-meter-analytics) | IoT smart meter revenue intelligence |

---

*Built from real operational experience as a Data Analyst in the electricity distribution sector.*
