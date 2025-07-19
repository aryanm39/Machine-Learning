# Linear Regression Implementation using Scikit-learn
# Comprehensive examples covering height-weight prediction and salary prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =============================================================================
# 1. HEIGHT AND WEIGHT PREDICTION EXAMPLE
# =============================================================================

def height_weight_prediction():
    """
    Predicts weight based on height using a simple linear regression model
    """
    print("=== HEIGHT-WEIGHT PREDICTION EXAMPLE ===\n")
    
    # Sample heights in centimeters (input variable X)
    # Data should be in 2D array format for sklearn
    x = np.array([[160], [165], [170], [175], [180], [185], [190], [195], [200], [205]])
    
    # Corresponding weights in kilograms (output variable Y)
    # 1D array is acceptable for target variable
    y = np.array([55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    
    # Create and train the model
    regressor = LinearRegression()
    regressor.fit(x, y)
    
    # Make predictions for new heights
    predictions = [
        (175, regressor.predict([[175]])[0]),
        (183, regressor.predict([[183]])[0]),
        (156, regressor.predict([[156]])[0])
    ]
    
    # Display predictions
    for height, weight in predictions:
        print(f"Predicted weight for {height}cm height: {weight:.2f} kg")
    
    # Model evaluation
    accuracy_score = regressor.score(x, y) * 100
    slope = regressor.coef_[0]
    intercept = regressor.intercept_
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy_score:.2f}%")
    print(f"Coefficient (Slope): {slope:.2f}")
    print(f"Intercept: {intercept:.2f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Original Data', s=50)
    
    # Generate regression line
    y_predicted_line = regressor.predict(x)
    plt.plot(x, y_predicted_line, color='red', linewidth=2, label='Linear Regression Line')
    
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.title('Height vs Weight with Linear Regression Line')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return regressor, x, y

def perfectly_linear_example():
    """
    Demonstrates linear regression with perfectly linear data (100% accuracy)
    """
    print("\n=== PERFECTLY LINEAR DATA EXAMPLE ===\n")
    
    # Create perfectly linear data
    a = np.array([[10], [20], [30], [40], [50]])
    b = np.array([20, 40, 60, 80, 100])  # y = 2x
    
    # Create and train the model
    regressor_perfect = LinearRegression()
    regressor_perfect.fit(a, b)
    
    # Evaluate the model
    score_perfect = regressor_perfect.score(a, b) * 100
    print(f"Accuracy for perfectly linear data: {score_perfect:.2f}%")
    
    # Make a prediction
    predicted_perfect = regressor_perfect.predict([[23]])[0]
    print(f"Predicted value for input 23: {predicted_perfect:.2f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(a, b, color='blue', label='Original Data (Perfectly Linear)', s=50)
    plt.plot(a, regressor_perfect.predict(a), color='red', linewidth=2, 
             label='Linear Regression Line (Perfect)')
    
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Perfectly Linear Data Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return regressor_perfect

# =============================================================================
# 2. SALARY PREDICTION EXAMPLE (CSV DATA)
# =============================================================================

def create_sample_salary_data():
    """
    Creates sample salary data for demonstration purposes
    """
    # Create sample data similar to what would be in salary.csv
    np.random.seed(42)
    years_experience = np.arange(0.5, 11.5, 0.5)  # 0.5 to 11 years
    
    # Create salary with some linear relationship + noise
    base_salary = 30000
    salary_per_year = 8000
    noise = np.random.normal(0, 5000, len(years_experience))
    
    salary = base_salary + salary_per_year * years_experience + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'YearsExperience': years_experience,
        'Salary': salary
    })
    
    return data

def salary_prediction_example():
    """
    Predicts salary based on years of experience using train-test split
    """
    print("\n=== SALARY PREDICTION EXAMPLE ===\n")
    
    # Load data (using sample data for demonstration)
    data = create_sample_salary_data()
    
    # For real implementation, use:
    # data = pd.read_csv('salary.csv')
    
    # Display basic information about the dataset
    print(f"Data shape: {data.shape}")
    print(f"Column names: {data.columns.tolist()}")
    print(f"\nFirst few rows of data:")
    print(data.head())
    
    # Prepare the data
    x_salary = data['YearsExperience'].values.reshape(-1, 1)  # 2D array
    y_salary = data['Salary'].values  # 1D array
    
    print(f"\nShape of x_salary after reshape: {x_salary.shape}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        x_salary, y_salary, test_size=0.25, random_state=0
    )
    
    print(f"Length of X_train: {len(X_train)}")
    print(f"Length of X_test: {len(X_test)}")
    
    # Create and train the model
    regressor_salary = LinearRegression()
    regressor_salary.fit(X_train, y_train)
    
    # Evaluate model performance
    train_accuracy = regressor_salary.score(X_train, y_train) * 100
    test_accuracy = regressor_salary.score(X_test, y_test) * 100
    
    print(f"\nTraining data accuracy: {train_accuracy:.2f}%")
    print(f"Test data accuracy: {test_accuracy:.2f}%")
    
    # Make predictions on test set
    y_pred = regressor_salary.predict(X_test)
    
    # Display actual vs predicted values
    results_df = pd.DataFrame({
        'Actual Value': y_test,
        'Predicted Value': y_pred,
        'Difference': np.abs(y_test - y_pred)
    })
    
    print("\nComparison of Actual vs. Predicted Salaries for Test Data:")
    print(results_df.round(2))
    
    # Make predictions for new experience values
    new_predictions = [
        (20, regressor_salary.predict([[20]])[0]),
        (15, regressor_salary.predict([[15]])[0])
    ]
    
    print(f"\nPredictions for new experience values:")
    for years, salary in new_predictions:
        print(f"Predicted salary for {years} years experience: ${salary:,.2f}")
    
    # Display model coefficients
    salary_slope = regressor_salary.coef_[0]
    salary_intercept = regressor_salary.intercept_
    
    print(f"\nSalary Model Parameters:")
    print(f"Coefficient (Slope): {salary_slope:.2f}")
    print(f"Intercept: {salary_intercept:.2f}")
    
    # Calculate performance metrics
    calculate_performance_metrics(y_test, y_pred)
    
    # Visualize the results
    visualize_salary_prediction(X_train, y_train, X_test, y_test, regressor_salary)
    
    return regressor_salary, X_train, X_test, y_train, y_test, y_pred

def calculate_performance_metrics(y_test, y_pred):
    """
    Calculates and displays various performance metrics
    """
    print(f"\n=== PERFORMANCE METRICS ===")
    
    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:,.2f}")
    
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    
    # Average Prediction Error (in percentage)
    average_error_percentage = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print(f"Average Prediction Error (%): {average_error_percentage:.2f}%")
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")

def visualize_salary_prediction(X_train, y_train, X_test, y_test, regressor):
    """
    Creates comprehensive visualizations for salary prediction
    """
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training data with regression line
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Data')
    
    # Create smooth line for regression
    X_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_range_pred = regressor.predict(X_range)
    plt.plot(X_range, y_range_pred, color='red', linewidth=2, label='Regression Line')
    
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary ($)')
    plt.title('Training Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Test data with predictions
    plt.subplot(1, 3, 2)
    plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual Test Data')
    
    y_test_pred = regressor.predict(X_test)
    plt.scatter(X_test, y_test_pred, color='red', alpha=0.6, label='Predicted Values')
    
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary ($)')
    plt.title('Test Data: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals plot
    plt.subplot(1, 3, 3)
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, color='purple', alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run all linear regression examples
    """
    print("LINEAR REGRESSION WITH SCIKIT-LEARN\n")
    print("=" * 50)
    
    # Run height-weight prediction example
    regressor_hw, x_hw, y_hw = height_weight_prediction()
    
    # Run perfectly linear data example
    regressor_perfect = perfectly_linear_example()
    
    # Run salary prediction example
    regressor_salary, X_train, X_test, y_train, y_test, y_pred = salary_prediction_example()
    
    print(f"\n" + "=" * 50)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    return {
        'height_weight_model': regressor_hw,
        'perfect_linear_model': regressor_perfect,
        'salary_model': regressor_salary
    }

if __name__ == "__main__":
    models = main()