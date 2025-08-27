# Time Series Analysis and Forecasting with Facebook Prophet
# This code demonstrates sales forecasting for Furniture and Office Supplies categories

# =============================================================================
# INSTALLATION AND SETUP
# =============================================================================

# For pip users:
# sudo pip3 install fbprophet

# For Anaconda users:
# conda install fbprophet
# Note: Ensure plotly is also installed as a dependency

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fbprophet import Prophet

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

# Load data from Excel file
data = pd.read_excel('superstore.xls')

# Check original data shape
print(f"Original data shape: {data.shape}")

# =============================================================================
# FURNITURE DATA PROCESSING
# =============================================================================

# Filter data for Furniture category
furniture = data.loc[data['Category'] == 'Furniture']
print(f"Furniture data shape: {furniture.shape}")

# Drop unnecessary columns (keep only 'Order Date' and 'Sales')
cols_to_drop = [
    'Row ID', 'Order ID', 'Ship Mode', 'Customer ID', 'Customer Name', 
    'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 
    'Product ID', 'Category', 'Sub-Category', 'Product Name', 
    'Quantity', 'Discount', 'Profit'
]
furniture.drop(columns=cols_to_drop, axis=1, inplace=True)

# Sort by Order Date
furniture.sort_values(by='Order Date', inplace=True)

# Group by Order Date and sum Sales, then reset index
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

# Set Order Date as index
furniture.set_index('Order Date', inplace=True)

# Resample to monthly mean (MS = Month Start)
y = furniture['Sales'].resample('MS').mean()

# Example: Access data for specific year
print("2017 Furniture Sales:")
print(y['2017'])

# =============================================================================
# OFFICE SUPPLIES DATA PROCESSING
# =============================================================================

# Filter data for Office Supplies category
office = data.loc[data['Category'] == 'Office Supplies']

# Sort by Order Date
office.sort_values(by='Order Date', inplace=True)

# Group by Order Date and sum Sales, then reset index
office = office.groupby('Order Date')['Sales'].sum().reset_index()

# Set Order Date as index
office.set_index('Order Date', inplace=True)

# Resample to monthly mean
y_office = office['Sales'].resample('MS').mean()

# =============================================================================
# PREPARE DATA FOR PROPHET AND COMPARISON
# =============================================================================

# Create DataFrames with Prophet-required column names ('ds' and 'y')
furniture_df = pd.DataFrame({'ds': y.index, 'y': y.values})
office_df = pd.DataFrame({'ds': y_office.index, 'y': y_office.values})

# Merge furniture and office data for comparison
store = furniture_df.merge(office_df, on='ds', how='inner')

# Rename columns after merge for clarity
store.rename(columns={'y_x': 'furniture_sales', 'y_y': 'office_sales'}, inplace=True)

# =============================================================================
# INITIAL TREND VISUALIZATION
# =============================================================================

# Plot furniture vs office sales trends
plt.figure(figsize=(20, 8))
plt.plot(store['ds'], store['furniture_sales'], label='Furniture Sales', color='blue')
plt.plot(store['ds'], store['office_sales'], label='Office Sales', color='red')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales of Furniture and Office Supplies')
plt.legend()
plt.show()

# =============================================================================
# IDENTIFY CROSSOVER POINTS
# =============================================================================

# Find first date when office sales exceeded furniture sales
office_higher_mask = store['office_sales'] > store['furniture_sales']
if office_higher_mask.any():
    first_crossover_idx = np.where(office_higher_mask)[0][0]
    first_date_office_higher = store.loc[first_crossover_idx, 'ds']
    print(f"First date office sales exceeded furniture sales: {first_date_office_higher}")

# =============================================================================
# PROPHET MODEL TRAINING - FURNITURE
# =============================================================================

# Create and configure Prophet model for furniture
furniture_model = Prophet(
    interval_width=0.95,
    weekly_seasonality=True,
    daily_seasonality=True
)

# Fit the model
furniture_model.fit(furniture_df)

# Create future dataframe for 24 months ahead
furniture_future = furniture_model.make_future_dataframe(periods=24, freq='MS')

# Generate forecast
furniture_forecast = furniture_model.predict(furniture_future)

# =============================================================================
# PROPHET MODEL TRAINING - OFFICE SUPPLIES
# =============================================================================

# Create and configure Prophet model for office supplies
office_model = Prophet(
    interval_width=0.95,
    weekly_seasonality=True,
    daily_seasonality=True
)

# Fit the model
office_model.fit(office_df)

# Create future dataframe for 24 months ahead
office_future = office_model.make_future_dataframe(periods=24, freq='MS')

# Generate forecast
office_forecast = office_model.predict(office_future)

# =============================================================================
# INDIVIDUAL FORECAST VISUALIZATION
# =============================================================================

# Plot furniture sales forecast
furniture_model.plot(
    furniture_forecast,
    figsize=(18, 6),
    xlabel='Date',
    ylabel='Sales',
    title='Furniture Sales Forecast'
)
plt.show()

# Plot office supplies sales forecast
office_model.plot(
    office_forecast,
    figsize=(18, 6),
    xlabel='Date',
    ylabel='Sales',
    title='Office Supplies Sales Forecast'
)
plt.show()

# =============================================================================
# COMBINED FORECAST ANALYSIS
# =============================================================================

# Create copies for manipulation
furniture_forecast_copy = furniture_forecast.copy()
office_forecast_copy = office_forecast.copy()

# Rename columns for easier identification after merge
furniture_columns = {
    'ds': 'Date',
    'yhat': 'Furniture_yhat',
    'yhat_lower': 'Furniture_yhat_lower',
    'yhat_upper': 'Furniture_yhat_upper',
    'trend': 'Furniture_trend',
    'trend_lower': 'Furniture_trend_lower',
    'trend_upper': 'Furniture_trend_upper'
}

office_columns = {
    'ds': 'Date',
    'yhat': 'Office_yhat',
    'yhat_lower': 'Office_yhat_lower',
    'yhat_upper': 'Office_yhat_upper',
    'trend': 'Office_trend',
    'trend_lower': 'Office_trend_lower',
    'trend_upper': 'Office_trend_upper'
}

furniture_forecast_copy.rename(columns=furniture_columns, inplace=True)
office_forecast_copy.rename(columns=office_columns, inplace=True)

# Merge forecasted dataframes
forecast_combined = pd.merge(
    furniture_forecast_copy, 
    office_forecast_copy, 
    on='Date', 
    how='inner'
)

# =============================================================================
# COMBINED FORECAST VISUALIZATION
# =============================================================================

# Plot combined trends
plt.figure(figsize=(10, 7))
plt.plot(
    forecast_combined['Date'], 
    forecast_combined['Furniture_trend'], 
    label='Furniture Trend', 
    color='blue'
)
plt.plot(
    forecast_combined['Date'], 
    forecast_combined['Office_trend'], 
    label='Office Trend', 
    color='red'
)
plt.xlabel('Date')
plt.ylabel('Sale')
plt.title('Furniture vs. Office Supply Sale Trends')
plt.legend()
plt.show()

# Plot combined yhat values (predicted values)
plt.figure(figsize=(10, 7))
plt.plot(
    forecast_combined['Date'], 
    forecast_combined['Furniture_yhat'], 
    label='Furniture Yhat', 
    color='blue'
)
plt.plot(
    forecast_combined['Date'], 
    forecast_combined['Office_yhat'], 
    label='Office Yhat', 
    color='red'
)
plt.xlabel('Date')
plt.ylabel('Sale')
plt.title('Furniture vs. Office Supply Sale (Yhat)')
plt.legend()
plt.show()

# =============================================================================
# MODEL COMPONENTS ANALYSIS
# =============================================================================

# Plot furniture model components (trend, seasonality, etc.)
furniture_model.plot_components(furniture_forecast)
plt.show()

# Plot office supplies model components
office_model.plot_components(office_forecast)
plt.show()

print("Forecasting analysis complete!")