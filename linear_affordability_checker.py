import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load all years for income and housing
years = ['2019', '2020', '2021', '2022', '2023']
dp03 = {year: pd.read_csv(f"dp03_cleaned/dp03_{year}.csv") for year in years}
dp04 = {year: pd.read_csv(f"dp04_cleaned/dp04_{year}.csv") for year in years}

# Ensure ZIP codes are strings with 5 digits
for df in dp03.values():
    df["zip"] = df["zip"].astype(str).str.zfill(5)
for df in dp04.values():
    df["zip"] = df["zip"].astype(str).str.zfill(5)

def run_affordability_check():
    print("Housing Affordability Checker")
    zip_input = input("Enter a 5-digit ZIP code: ").strip().zfill(5)
    housing_type = input("Are you looking to rent or own? (rent/own): ").strip().lower()
    user_income = input("Enter your annual income (or press Enter to use ZIP's median): ").strip()

    income_trend = []
    cost_trend = []
    ratio_trend = []

    for year in years:
        income_df = dp03[year]
        cost_df = dp04[year]

        income_row = income_df[income_df["zip"] == zip_input]
        cost_row = cost_df[cost_df["zip"] == zip_input]

        if income_row.empty or cost_row.empty:
            continue

        # Use ZIP median or user-provided income
        if user_income:
            try:
                income = float(user_income)
            except ValueError:
                print("Invalid income entered.")
                return
        else:
            income_str = str(income_row.iloc[0]["median_income_dollars"])
            income_str = income_str.replace(",", "").replace("+", "").strip()

            try:
                income = float(income_str)
            except ValueError:
                print(f"Could not interpret income value: {income_str}")
                return


        # Monthly housing cost
        if housing_type == "rent":
            cost = cost_row.iloc[0].get("median_rent_dollars")
        elif housing_type == "own":
            cost = cost_row.iloc[0].get("median_owner_cost_dollars")
        else:
            print("Invalid housing type. Choose 'rent' or 'own'.")
            return

        try:
            cost = float(cost)
        except:
            continue  # skip year if cost is missing

        annual_cost = cost * 12
        ratio = (annual_cost / income) * 100

        income_trend.append((year, income))
        cost_trend.append((year, annual_cost))
        ratio_trend.append((year, ratio))

    if not ratio_trend:
        print("No data available for that ZIP code across any years.")
        return

    print("\nAffordability Trend by Year")
    print(f"{'Year':<6}{'Income':>12}{'Cost':>12}{'Ratio (%)':>12}")
    for year, income in income_trend:
        cost = dict(cost_trend)[year]
        ratio = dict(ratio_trend)[year]
        print(f"{year:<6}${income:>11,.0f}${cost:>11,.0f}{ratio:>12.1f}")
        
    # Latest year evaluation
    latest_year = ratio_trend[-1][0]
    latest_ratio = ratio_trend[-1][1]
    print(f"\nFor year {latest_year}, housing cost is {latest_ratio:.1f}% of income.")
    if latest_ratio > 30:
        print("This is considered UNAFFORDABLE housing.")
    else:
        print("This is considered affordable housing.")

    # Trend direction
    if len(ratio_trend) > 1:
        first, last = ratio_trend[0][1], ratio_trend[-1][1]
        if last > first:
            print("Affordability is worsening over time.")
        elif last < first:
            print("Affordability is improving over time.")
        else:
            print("Affordability has remained steady over time.")

    def predict_next_year_linear_trend(years_list, values):
        X = np.array(range(len(values))).reshape(-1, 1)
        y = np.array(values)
        model = LinearRegression().fit(X, y)
        next_value = model.predict([[len(values)]])[0]
        return next_value

    # Extract values from trends
    income_vals = [val for _, val in income_trend]
    cost_vals = [val for _, val in cost_trend]

    # Predict next year (2024)
    pred_income = predict_next_year_linear_trend(years, income_vals)
    pred_cost = predict_next_year_linear_trend(years, cost_vals)
    pred_ratio = (pred_cost / pred_income) * 100

    print("\n--- Forecast for Next Year (2024) ---")
    print(f"Predicted Income: ${pred_income:,.0f}")
    print(f"Predicted Housing Cost: ${pred_cost:,.0f}")
    print(f"Predicted Housing Cost Ratio: {pred_ratio:.1f}%")

    if pred_ratio > 30:
        print("Prediction: UNAFFORDABLE in 2024.")
    else:
        print("Prediction: Affordable in 2024.")

    # --- 10-Year Risk Assessment: 2024–2033 --- #
    print("\n--- 10-Year Risk Assessment ---")

    # Step 1: Calculate average income growth from ZIP trend
    income_diffs = np.diff(income_vals)
    income_growth_rates = income_diffs / income_vals[:-1]
    avg_income_growth = np.nanmean(income_growth_rates)

    print(f"Average ZIP income growth rate: {avg_income_growth*100:.2f}%")

    # Step 2: Forecast housing cost using linear regression
    cost_model = LinearRegression().fit(
        np.array(range(len(cost_vals))).reshape(-1, 1),
        np.array(cost_vals)
    )

    # Step 3: Set initial income for projection
    start_income = float(user_income) if user_income.strip() else income_vals[-1]

    # Step 4: Project future years
    future_years = [int(years[-1]) + i + 1 for i in range(10)]
    future_incomes = [start_income * (1 + avg_income_growth) ** i for i in range(1, 11)]
    future_costs = [cost_model.predict([[len(cost_vals) + i]])[0] for i in range(10)]
    future_ratios = [(cost / income) * 100 for cost, income in zip(future_costs, future_incomes)]

    # Step 5: Determine if affordability is crossed
    unaffordable_year = None
    for i, ratio in enumerate(future_ratios):
        if ratio > 30:
            unaffordable_year = future_years[i]
            break

    # Step 6: Print forecast results
    if unaffordable_year:
        idx = future_years.index(unaffordable_year)
        print(f"Warning: This ZIP code is on track to become UNAFFORDABLE by {unaffordable_year}.")
        print(f"Predicted values for {unaffordable_year}:")
        print(f"  Income: ${future_incomes[idx]:,.0f}")
        print(f"  Housing Cost: ${future_costs[idx]:,.0f}")
        print(f"  Housing Cost Ratio: {future_ratios[idx]:.1f}%")
    else:
        print("No risk of becoming unaffordable in the next 10 years based on current trends.")
        print(f"Predicted values for 2033:")
        print(f"  Income: ${future_incomes[-1]:,.0f}")
        print(f"  Housing Cost: ${future_costs[-1]:,.0f}")
        print(f"  Housing Cost Ratio: {future_ratios[-1]:.1f}%")




    # Latest year evaluation
    latest_year = ratio_trend[-1][0]
    latest_ratio = ratio_trend[-1][1]
    print(f"\nFor year {latest_year}, housing cost is {latest_ratio:.1f}% of income.")
    if latest_ratio > 30:
        print("This is considered UNAFFORDABLE housing.")
    else:
        print("This is considered affordable housing.")

    # Trend direction
    if len(ratio_trend) > 1:
        first, last = ratio_trend[0][1], ratio_trend[-1][1]
        if last > first:
            print("Affordability is worsening over time.")
        elif last < first:
            print("Affordability is improving over time.")
        else:
            print("Affordability has remained steady over time.")

if __name__ == "__main__":
    run_affordability_check()