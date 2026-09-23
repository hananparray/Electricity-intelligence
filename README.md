# Electricity Consumption & Energy Production Intelligence

## 1. Project Overview

**Electricity Consumption & Energy Production Intelligence** is a data analytics and machine-learning decision-support dashboard built with Python and Streamlit.

The project analyzes hourly electricity consumption and production data to answer:

- How does electricity demand change over time?
- When are peak-demand periods?
- How is electricity generation distributed across energy sources?
- What is the renewable vs non-renewable generation mix?
- When does production exceed or fall below consumption?
- Can historical demand patterns be used to forecast future consumption?

The project follows the mentor's recommended flow:

**Data → Information → Insights → Decision → Action**

## 2. Objectives

1. Clean and validate the hourly electricity dataset.
2. Analyze consumption patterns by hour, day, month, and year.
3. Analyze electricity generation by source.
4. Compare renewable and non-renewable generation.
5. Study the production-consumption energy balance.
6. Build a simple electricity demand forecasting model.
7. Present KPIs, trends, insights, and recommendations in an interactive dashboard.

## 3. Dataset

**Dataset:** Hourly Electricity Consumption & Production

**Source:** Kaggle

**Dataset link:** https://www.kaggle.com/datasets/mayuringle8890/electricity

The dataset contains 46,011 raw hourly records and these fields:

- `DateTime`
- `Consumption`
- `Production`
- `Nuclear`
- `Wind`
- `Hydroelectric`
- `Oil and Gas`
- `Coal`
- `Solar`
- `Biomass`

The source-level generation columns allow analysis of the energy mix in addition to total consumption and production.

## 4. Data Cleaning

The application:

- Converts `DateTime` to datetime format.
- Removes exact duplicate rows.
- Aggregates repeated timestamps by taking the mean of numerical observations.
- Creates time features such as hour, day of week, month, and year.
- Keeps the supplied `Production` column as the reported total.
- Clips negative Wind values to zero **only for source-mix calculations** because generation cannot be negative.
- Calculates:
  - `Renewable = Wind + Hydroelectric + Solar + Biomass`
  - `NonRenewable = Nuclear + Oil and Gas + Coal`
  - `EnergyBalance = Production - Consumption`

The raw `Wind` column is not overwritten.

## 5. Analytics

### Consumption
- Total consumption
- Average consumption
- Peak consumption
- Lowest consumption
- Hourly demand profile
- Weekday/weekend comparison
- Monthly trends

### Energy Production
- Total production
- Source-wise generation
- Energy-source contribution
- Renewable vs non-renewable generation
- Daily source trends

### Energy Balance
- Production minus consumption
- Surplus periods
- Deficit periods

## 6. Machine Learning

The optional ML component is **electricity demand forecasting**.

### Target
`Consumption`

### Features
- Hour
- Day of week
- Month
- Day of year
- 1-hour lag
- 24-hour lag
- 168-hour lag
- 24-hour rolling average
- 168-hour rolling average

### Model
**Random Forest Regressor**

A chronological **80/20 train-test split** is used so that future observations are not used to train the model.

### Evaluation
The model is evaluated using:

- MAE
- RMSE
- R²

On the uploaded dataset after cleaning, the model produced approximately:

- **MAE:** 95.54 MWh
- **RMSE:** 128.46 MWh
- **R²:** 0.981

## 7. Key Findings from the Current Dataset

- Recorded peak consumption: **9,615 MWh** at **2021-01-19 10:00:00**.
- Recorded minimum consumption: **3,889 MWh** at **2023-06-04 06:00:00**.
- Highest average-demand hour: **20:00**.
- Lowest average-demand hour: **03:00**.
- Average weekday consumption: **6,810 MWh**.
- Average weekend consumption: **6,033 MWh**.
- Source-level generation was approximately **43.9% renewable** under the project's renewable-source definition.

## 8. Dashboard

The Streamlit application contains five sections:

1. **Overview** — KPIs and high-level consumption/energy balance.
2. **Consumption Analysis** — hourly, weekday, and monthly demand patterns.
3. **Energy Production** — generation by source and energy mix.
4. **Demand Forecast** — model metrics, test-period prediction, and future forecast.
5. **Insights & Data Quality** — key findings and preprocessing notes.

## 9. How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit

```bash
streamlit run electricity_intelligence.py
```

Then upload `Electricity.csv` in the sidebar.

For local execution, the CSV may also be placed in the same folder as the Python file.

## 10. Limitations

- The dataset does not contain external variables such as temperature, weather, population, price, or industrial activity.
- Therefore, the project focuses on temporal patterns and recorded production variables rather than claiming causal effects from external factors.
- The source dataset contains some negative Wind observations and differences between the supplied Production total and source-level generation sums; these are explicitly handled/documented rather than silently ignored.
- Forecast accuracy depends on the historical patterns represented in the dataset.

## 11. Future Scope

- Add weather and temperature data.
- Add regional/state-level data.
- Compare additional forecasting algorithms.
- Add anomaly detection.
- Add a more advanced forecasting model if required.
- Connect to a live electricity data source.

## 12. Technologies

Python, Pandas, NumPy, Plotly, Scikit-learn, Streamlit.

## 13. Project Structure

```text
electricity-intelligence/
├── electricity_intelligence.py
├── requirements.txt
└── README.md
```

The project report is submitted separately as required by the internship.
