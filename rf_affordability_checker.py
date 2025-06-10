"""
This script evaluates housing affordability for a given ZIP code using a Random Forest model.
It combines 5 years of U.S. Census ACS data (2019–2023) from dp02, dp03, and dp04 to predict
whether the housing cost in a ZIP is affordable (≤ 30% of income) or unaffordable. The user
can input a ZIP code and income, and the model returns affordability classification.

New: The script also generates 10-year forward predictions of affordability ratio trends and
estimates whether a ZIP code is at risk of becoming unaffordable in the future. It now includes
predicted local income growth rates to adjust user-entered salary projections more accurately.

Newer: Each ZIP is graded on how favorably income growth compares to rent increases using
percentile ranks across all ZIPs. This helps evaluate the long-term quality of settling down.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ----------- CONFIG ----------- #
YEARS = ['2019', '2020', '2021', '2022', '2023']
HOUSING_TYPE_COL = {
    'rent': 'median_rent_dollars',
    'own': 'median_owner_cost_dollars'
}
FEATURE_COLS = [
    "income", "cost", "total_units_count", "occupied_units_count",
    "vacant_units_count", "computer_access_percent",
    "management_business_science_arts_jobs_count"
]
# ------------------------------ #

def load_data():
    dp02 = pd.concat([pd.read_csv(f"dp02_cleaned/dp02_{year}.csv") for year in YEARS])
    dp03 = pd.concat([pd.read_csv(f"dp03_cleaned/dp03_{year}.csv") for year in YEARS])
    dp04 = pd.concat([pd.read_csv(f"dp04_cleaned/dp04_{year}.csv") for year in YEARS])

    for df in [dp02, dp03, dp04]:
        df["zip"] = df["zip"].astype(str).str.zfill(5)

    return dp02, dp03, dp04

def prepare_dataset(dp02, dp03, dp04, housing_type):
    df = dp02.merge(dp03, on=["zip", "year"], suffixes=('', '_dp03'))
    df = df.merge(dp04, on=["zip", "year"], suffixes=('', '_dp04'))
    df = df.copy()

    df["income"] = pd.to_numeric(df["median_income_dollars"].replace(r'[\$,]', '', regex=True), errors='coerce')
    cost_col = HOUSING_TYPE_COL[housing_type]
    df["cost"] = pd.to_numeric(df[cost_col].replace(r'[\$,]', '', regex=True), errors='coerce')

    df = df.dropna(subset=["income", "cost"])

    df["housing_ratio"] = (df["cost"] * 12) / df["income"] * 100
    df["affordable"] = (df["housing_ratio"] <= 30).astype(int)

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    return df

def get_features_and_labels(df):
    X = df[FEATURE_COLS]
    y = df["affordable"]
    return X, y

def get_income_growth_rate(df, zip_code):
    zip_data = df[df["zip"] == zip_code].sort_values("year")
    income_vals = zip_data["income"].values
    if len(income_vals) < 2:
        return 0.03
    pct_changes = np.diff(income_vals) / income_vals[:-1]
    return np.nanmean(pct_changes)

def grade_location(ratio, all_ratios):
    percentile = (np.sum(all_ratios <= ratio) / len(all_ratios)) * 100
    if percentile >= 97: return "A+"
    elif percentile >= 93: return "A"
    elif percentile >= 90: return "A-"
    elif percentile >= 87: return "B+"
    elif percentile >= 83: return "B"
    elif percentile >= 80: return "B-"
    elif percentile >= 70: return "C+"
    elif percentile >= 60: return "C"
    elif percentile >= 50: return "C-"
    elif percentile >= 40: return "D+"
    elif percentile >= 30: return "D"
    elif percentile >= 20: return "D-"
    else: return "F"

def run_console_app(model, full_data, housing_type):
    print("Housing Affordability Risk Prediction (Random Forest)")

    zip_input = input("Enter a 5-digit ZIP code: ").strip().zfill(5)
    income_input = input("Enter your annual income (or press Enter to use ZIP median): ").strip()

    row_hist = full_data[full_data["zip"] == zip_input].sort_values("year")
    if row_hist.empty:
        print("No data found for that ZIP code.")
        return

    row = row_hist.iloc[-1:]

    if income_input:
        try:
            user_income = float(income_input)
        except ValueError:
            print("Invalid income format.")
            return
    else:
        user_income = row.iloc[0]["income"]

    cost = row.iloc[0]["cost"]
    ratio = (cost * 12) / user_income * 100

    row_features = row[FEATURE_COLS].copy()
    row_features["income"] = user_income
    row_features["cost"] = cost
    row_features = row_features.fillna(0)

    prediction = model.predict(row_features.values.reshape(1, -1))[0]

    print(f"\nZIP Code: {zip_input}")
    print(f"Annual Income: ${user_income:,.0f}")
    print(f"Annual Housing Cost: ${cost * 12:,.0f}")
    print(f"Housing Cost Ratio: {ratio:.1f}%")
    print("Prediction:", "AFFORDABLE" if prediction == 1 else "UNAFFORDABLE")

    print("\nAffordability Trend by Year")
    print(f"{'Year':<6}{'Income':>12}{'Cost':>12}{'Ratio (%)':>12}")
    for _, row in row_hist.iterrows():
        yr = int(row['year'])
        inc = row['income']
        cst = row['cost']
        rat = (cst * 12) / inc * 100
        print(f"{yr:<6}${inc:>11,.0f}${cst*12:>11,.0f}{rat:>12.1f}")

    print("\n--- 10-Year Risk Forecast ---")
    income_base = row_hist["income"].values
    cost_base = row_hist["cost"].values
    if len(income_base) < 2:
        print("Insufficient history for trend analysis.")
        return

    income_trend = np.polyfit(range(len(income_base)), income_base, 1)
    cost_trend = np.polyfit(range(len(cost_base)), cost_base, 1)
    forecast_years = list(range(len(income_base), len(income_base)+10))
    forecast_cost = np.polyval(cost_trend, forecast_years)

    if income_input:
        growth_rate = get_income_growth_rate(row_hist, zip_input)
        forecast_income = np.array([user_income * (1 + growth_rate)**i for i in range(1, 11)])
        income_change = ((forecast_income[-1] - user_income) / user_income) * 100
        print(f"\nBased on ZIP {zip_input}, projected income increase over 10 years: {income_change:.1f}%")
    else:
        forecast_income = np.polyval(income_trend, forecast_years)
        income_change = ((forecast_income[-1] - forecast_income[0]) / forecast_income[0]) * 100
        print(f"\nBased on ZIP {zip_input}, projected median income increase over 10 years: {income_change:.1f}%")

    cost_change = ((forecast_cost[-1] - forecast_cost[0]) / forecast_cost[0]) * 100
    print(f"Projected housing cost increase over 10 years: {cost_change:.1f}%")

    forecast_ratio = (forecast_cost * 12) / forecast_income * 100

    unaffordable_year = None
    for i, ratio in enumerate(forecast_ratio):
        if ratio > 30:
            unaffordable_year = int(row_hist.iloc[-1]['year']) + i + 1
            print(f"\nWarning: Expected to become UNAFFORDABLE by {unaffordable_year}.")
            print(f"Predicted values for {unaffordable_year}:")
            print(f"  Income: ${forecast_income[i]:,.0f}")
            print(f"  Housing Cost: ${forecast_cost[i]*12:,.0f}")
            print(f"  Housing Cost Ratio: {forecast_ratio[i]:.1f}%")
            break

    if not unaffordable_year:
        print("\nNo risk of becoming unaffordable in the next 10 years based on current trends.")
        print(f"Predicted values for 2033:")
        print(f"  Income: ${forecast_income[-1]:,.0f}")
        print(f"  Housing Cost: ${forecast_cost[-1]*12:,.0f}")
        print(f"  Housing Cost Ratio: {forecast_ratio[-1]:.1f}%")

    zip_ratios = []
    for zip_code in full_data['zip'].unique():
        hist = full_data[full_data['zip'] == zip_code].sort_values("year")
        if len(hist) >= 2:
            i_vals = hist['income'].values
            c_vals = hist['cost'].values
            i_trend = np.polyfit(range(len(i_vals)), i_vals, 1)
            c_trend = np.polyfit(range(len(c_vals)), c_vals, 1)
            i_proj = np.polyval(i_trend, [len(i_vals), len(i_vals)+9])
            c_proj = np.polyval(c_trend, [len(c_vals), len(c_vals)+9])
            income_change_zip = (i_proj[-1] - i_proj[0]) / i_proj[0] * 100
            cost_change_zip = (c_proj[-1] - c_proj[0]) / c_proj[0] * 100
            if cost_change_zip > 0:
                zip_ratios.append(income_change_zip / cost_change_zip)

    if cost_change > 0:
        user_ratio = income_change / cost_change
        user_grade = grade_location(user_ratio, np.array(zip_ratios))
        print(f"\nZIP Code {zip_input} receives a housing opportunity grade of: {user_grade}")
        print(f"  (Ratio of income to housing cost growth: {user_ratio:.2f})")

if __name__ == "__main__":
    housing_type = input("Are you evaluating for rent or own? ").strip().lower()
    if housing_type not in HOUSING_TYPE_COL:
        print("Invalid housing type. Use 'rent' or 'own'.")
        exit()

    dp02, dp03, dp04 = load_data()
    full_data = prepare_dataset(dp02, dp03, dp04, housing_type)
    X, y = get_features_and_labels(full_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("\nModel Evaluation:")
    print(classification_report(y_test, model.predict(X_test)))

    run_console_app(model, full_data, housing_type)