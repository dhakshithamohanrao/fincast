# FinCast

FinCast is a personal finance app. You import your bank transactions as a CSV, and it:

- sorts them into categories automatically,
- forecasts next month's income, spending and end balance, with a likely range,
- flags unusual transactions (for example a possible double charge),
- tracks savings goals and shows how a change in spending would affect your savings.

## Results

All numbers come from a 12-month rolling backtest on synthetic data (300 users).

| Test | Model | Baseline (same month last year) |
|---|---|---|
| Variable spending, MAE (seen users) | 53.76 | 65.56 |
| Variable spending, MAE (new users) | 52.64 | 62.81 |
| Users with under 12 months of history | 61.59 | 65.56 |

- The model must beat the baseline by at least 10% to be used. It does, in both tests.
- The 80% forecast range covers the real value about 83% of the time.
- Rent and subscriptions are copied from last month, because they do not change.



Register an account, then upload a bank CSV on the Import page. A sample file is in `data/sample/`.



Commit the code and the retrained files in `models/` together, so they always agree.

## Known limits

- The models are trained on synthetic data. Real bank data may behave differently.
- Savings so far are estimated from the imported transactions, because the real balance is unknown.
- The 61.59 result for short histories is a best case: that test only fills the missing "same month last year" value.
- The currency setting is saved but not yet used for number formatting.
- Forecasts are for education only, not financial advice.

## Commands

Run the app:

```
pip install -r requirements.txt
streamlit run dashboard/Home.py
```

Retrain the models and run the tests:

```
python -m src.training.train_forecaster
python -m pytest tests -q
```

## Screenshots

### 1. Home
![Home](docs/screenshots/01_home.png)

### 2. Overview
![Overview](docs/screenshots/02_overview.png)

### 3. Import transactions
![Import](docs/screenshots/03_import.png)

### 4. Forecast
![Forecast](docs/screenshots/04_forecast.png)
![Forecast by category](docs/screenshots/04b_forecast_categories.png)

### 5. What if
![What if](docs/screenshots/05_what_if.png)

### 6. Anomalies
![Anomalies](docs/screenshots/06_anomalies.png)
![Anomalies after review](docs/screenshots/06b_anomalies_reviewed.png)

### 7. Savings goals
![Goals](docs/screenshots/07_goals.png)

### 8. Profile
![Profile](docs/screenshots/08_profile.png)
![Profile account](docs/screenshots/08b_profile_account.png)
