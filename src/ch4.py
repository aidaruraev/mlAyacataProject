import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


def load_and_clean_data(filepath):
    """
    Load data from a CSV file and clean the column names by removing spaces.
    """
    df = pd.read_csv(filepath, encoding="latin")
    df.columns = df.columns.str.replace(' ', '')
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    return df


def create_timeseries_df(df):
    """
    Create a time-series DataFrame by resampling data into monthly frequency.
    """
    ts_df = df[["OrderID", "OrderDate", "Quantity", "Sales", "Category"]].copy()
    ts_df.set_index("OrderDate", inplace=True) # inplace=True: This argument means that the operation modifies the original DataFrame
    return ts_df


def compute_resampled_metrics(ts_df):
    """
    Compute resampled metrics such as unique orders, quantities, and sales changes.
    """
    monthly_unique_orders = ts_df["OrderID"].resample("MS").nunique()
    monthly_unique_order_changes = (
        monthly_unique_orders - monthly_unique_orders.shift()
    ) / monthly_unique_orders.shift() * 100

    monthly_quantities = ts_df["Quantity"].resample("MS").sum()
    monthly_quantities_changes = (
        monthly_quantities - monthly_quantities.shift()
    ) / monthly_quantities.shift() * 100

    monthly_sales = ts_df["Sales"].resample("MS").sum()
    monthly_sales_changes = (
        monthly_sales - monthly_sales.shift()
    ) / monthly_sales.shift() * 100

    return monthly_unique_order_changes, monthly_quantities_changes, monthly_sales_changes


def compute_moving_averages(monthly_sales, window_6=6, window_12=12):
    """
    Compute 6-month and 12-month rolling averages for sales data.
    """
    m6_ma_sales = monthly_sales.rolling(window_6).mean()
    m12_ma_sales = monthly_sales.rolling(window_12).mean()
    return m6_ma_sales, m12_ma_sales


def decompose_and_analyze_sales(furniture_sales, model='additive'): # 2 types of model: additive and multiplicative
    """
    Decompose the furniture sales data into trend, seasonality, and residuals.
    """
    decomposition = sm.tsa.seasonal_decompose(furniture_sales, model=model)
    trend_seasonal = decomposition.trend + decomposition.seasonal
    return decomposition, trend_seasonal


def calculate_similarity_metrics(original_sales, reconstructed_sales):
    """
    Calculate correlation, Euclidean distance, and RMSE between original and reconstructed sales data.
    """
    mask = ~np.isnan(original_sales) & ~np.isnan(reconstructed_sales)

    corr = np.corrcoef(list(original_sales[mask]), list(reconstructed_sales[mask]))[0, 1]
    dist = np.sqrt(np.square(original_sales[mask] - reconstructed_sales[mask]).sum())
    rmse = np.sqrt(np.square(original_sales[mask] - reconstructed_sales[mask]).mean())

    return corr, dist, rmse


def forecast_with_arima(sales_data, order=(12, 1, 3), steps=6):
    """
    Forecast sales using the ARIMA model.
    """
    model = ARIMA(sales_data, order=order)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=steps)
    return forecast


def forecast_with_prophet(sales_data, future_periods=24):
    """
    Forecast sales using the Prophet model.
    """
    # Prophet requires the input DataFrame to have specific column names: ds for the date/time and y for the value you want to forecast.
    sales_df = pd.DataFrame(sales_data).reset_index()
    sales_df.columns = ["ds", "y"]

    model = Prophet()
    model.fit(sales_df)

    future_dates = model.make_future_dataframe(periods=future_periods, freq='MS')
    forecast = model.predict(future_dates)

    return forecast


# Main workflow
df = load_and_clean_data("../data/data4.csv")
ts_df = create_timeseries_df(df)
monthly_sales = ts_df["Sales"].resample("MS").sum()

# Decompose and analyze sales
furniture_sales = ts_df.loc[ts_df["Category"] == "Furniture"]["Sales"].resample("MS").sum()
decomposition, reconstructed_sales = decompose_and_analyze_sales(furniture_sales)

# Calculate similarity metrics
corr, dist, rmse = calculate_similarity_metrics(furniture_sales, reconstructed_sales)
print(f"Correlation: {corr:.02f}\nEuclidean Distance: {dist:.02f}\nRMSE: {rmse:.02f}")

# Forecast with ARIMA
arima_forecast = forecast_with_arima(furniture_sales[:"2017-06-01"])
print(arima_forecast.predicted_mean)

# Forecast with Prophet
prophet_forecast = forecast_with_prophet(furniture_sales[:"2016-12-01"])
print(prophet_forecast)