# Regression Techniques in Python
# Demonstrating various regression techniques with sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# =============================================================================
# 1. SIMPLE LINEAR REGRESSION
# =============================================================================

def simple_linear_regression():
    """
    Demonstrates simple linear regression for accuracy check
    """
    # Load data (example assumes salary_experience.csv exists)
    # data = pd.read_csv('salary_experience.csv')
    # x = data['Experience'].values.reshape(-1, 1)  # Reshape for sklearn
    # y = data['Salary'].values
    
    # For demonstration, creating sample data
    np.random.seed(42)
    x = np.random.rand(100, 1) * 10  # Experience (0-10 years)
    y = 30000 + 5000 * x.flatten() + np.random.normal(0, 2000, 100)  # Salary
    
    # Create and fit the model
    regressor = LinearRegression()
    regressor.fit(x, y)
    
    # Calculate accuracy
    accuracy = np.round(regressor.score(x, y) * 100, 3)
    print(f"Linear Regression Accuracy: {accuracy}%")
    
    return x, y, regressor

# =============================================================================
# 2. POLYNOMIAL REGRESSION
# =============================================================================

def polynomial_regression(x, y, degree=3):
    """
    Demonstrates polynomial regression with specified degree
    """
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    xp = poly.fit_transform(x)
    
    # Fit the model
    regressor = LinearRegression()
    regressor.fit(xp, y)
    
    # Calculate accuracy
    accuracy_poly = np.round(regressor.score(xp, y) * 100, 3)
    print(f"Polynomial Regression (Degree {degree}) Accuracy: {accuracy_poly}%")
    
    return xp, regressor, poly

# =============================================================================
# 3. PLOTTING POLYNOMIAL REGRESSION
# =============================================================================

def plot_polynomial_regression(x, y, xp, regressor):
    """
    Visualizes polynomial regression curve
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of original data
    plt.scatter(x, y, color='blue', alpha=0.6, label='Original Data')
    
    # Sort data for smooth curve plotting
    sort_idx = np.argsort(x.flatten())
    x_sorted = x[sort_idx]
    xp_sorted = xp[sort_idx]
    
    # Plot regression curve
    plt.plot(x_sorted, regressor.predict(xp_sorted), 
             color='red', linewidth=2, label='Polynomial Regression Curve')
    
    plt.xlabel('Experience')
    plt.ylabel('Salary')
    plt.title('Polynomial Regression Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# =============================================================================
# 4. FINDING OPTIMAL POLYNOMIAL DEGREE
# =============================================================================

def find_optimal_degree(x, y, max_degree=30):
    """
    Finds the optimal polynomial degree by testing different degrees
    """
    accuracy_list = []
    
    # Loop through degrees from 2 to max_degree
    for n in range(2, max_degree + 1):
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=n)
        xp = poly_features.fit_transform(x)
        
        # Fit the model
        regressor = LinearRegression()
        regressor.fit(xp, y)
        
        # Calculate accuracy
        accuracy = np.round(regressor.score(xp, y) * 100, 3)
        accuracy_list.append(accuracy)
    
    # Find optimal degree
    max_accuracy = max(accuracy_list)
    optimal_degree_index = accuracy_list.index(max_accuracy)
    optimal_degree = optimal_degree_index + 2  # Add 2 because loop started from degree 2
    
    print(f"Highest accuracy achieved: {max_accuracy}%")
    print(f"Optimal Polynomial Degree: {optimal_degree}")
    
    # Plot accuracy vs degree
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_degree + 1), accuracy_list, 
             color='red', marker='o', label='Accuracy per Degree')
    plt.scatter(optimal_degree, max_accuracy, 
                color='blue', s=100, zorder=5, label='Max Accuracy Point')
    
    plt.xlabel('Degree')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Polynomial Degree')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return optimal_degree, max_accuracy

# =============================================================================
# 5. FINAL MODEL WITH OPTIMAL DEGREE
# =============================================================================

def final_polynomial_model(x, y, optimal_degree):
    """
    Creates final model with optimal polynomial degree
    """
    # Create polynomial features with optimal degree
    poly_features = PolynomialFeatures(degree=optimal_degree)
    xp = poly_features.fit_transform(x)
    
    # Fit the model
    regressor = LinearRegression()
    regressor.fit(xp, y)
    
    # Calculate final accuracy
    accuracy_final = np.round(regressor.score(xp, y) * 100, 3)
    print(f"Final Model Accuracy with Degree {optimal_degree}: {accuracy_final}%")
    
    return regressor, poly_features

# =============================================================================
# 6. MULTIPLE REGRESSION
# =============================================================================

def multiple_regression():
    """
    Demonstrates multiple regression with multiple input features
    """
    # Load data (using mtcars dataset as example)
    # cars_data = pd.read_csv('mtcars.csv')
    
    # For demonstration, creating sample data similar to mtcars
    np.random.seed(42)
    n_samples = 32
    
    data = {
        'disp': np.random.normal(230, 120, n_samples),  # Displacement
        'hp': np.random.normal(146, 68, n_samples),     # Horsepower
        'wt': np.random.normal(3.2, 1.0, n_samples),   # Weight
    }
    
    # Create MPG based on relationships with other variables
    mpg = (40 - 0.02 * data['disp'] - 0.05 * data['hp'] - 
           3 * data['wt'] + np.random.normal(0, 2, n_samples))
    
    cars_data = pd.DataFrame(data)
    cars_data['mpg'] = mpg
    
    # Select features and target
    x_multi = cars_data[['disp', 'hp', 'wt']]
    y_multi = cars_data['mpg']
    
    # Create and fit the model
    regressor_multi = LinearRegression()
    regressor_multi.fit(x_multi, y_multi)
    
    # Calculate accuracy
    accuracy_multi = np.round(regressor_multi.score(x_multi, y_multi) * 100, 2)
    print(f"Multiple Regression Accuracy: {accuracy_multi}%")
    
    # Display coefficients and intercept
    coefficients = regressor_multi.coef_
    intercept = regressor_multi.intercept_
    
    print(f"Coefficients (disp, hp, wt): {coefficients}")
    print(f"Intercept: {intercept}")
    
    # Make a prediction
    new_input_values = np.array([[221, 102, 2.91]])
    predicted_mpg = regressor_multi.predict(new_input_values)
    print(f"Predicted MPG for input {new_input_values[0]}: {predicted_mpg[0]:.2f}")
    
    return x_multi, y_multi, regressor_multi

# =============================================================================
# 7. PLOTTING MULTIPLE REGRESSION RELATIONSHIPS
# =============================================================================

def plot_multiple_regression(x_multi, y_multi):
    """
    Visualizes relationships between individual features and target
    """
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Displacement vs. MPG
    plt.subplot(2, 2, 1)
    plt.scatter(x_multi['disp'], y_multi, color='red', alpha=0.6)
    plt.xlabel('Displacement')
    plt.ylabel('MPG')
    plt.title('Displacement vs. MPG')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Horsepower vs. MPG
    plt.subplot(2, 2, 2)
    plt.scatter(x_multi['hp'], y_multi, color='green', alpha=0.6)
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.title('Horsepower vs. MPG')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Weight vs. MPG
    plt.subplot(2, 2, 3)
    plt.scatter(x_multi['wt'], y_multi, color='purple', alpha=0.6)
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.title('Weight vs. MPG')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Leave empty or add correlation matrix
    plt.subplot(2, 2, 4)
    correlation_matrix = x_multi.corr()
    im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(im)
    plt.title('Feature Correlation Matrix')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to demonstrate all regression techniques
    """
    print("=== REGRESSION TECHNIQUES DEMONSTRATION ===\n")
    
    # 1. Simple Linear Regression
    print("1. Simple Linear Regression:")
    x, y, linear_regressor = simple_linear_regression()
    print()
    
    # 2. Polynomial Regression
    print("2. Polynomial Regression (Degree 3):")
    xp, poly_regressor, poly_features = polynomial_regression(x, y, degree=3)
    plot_polynomial_regression(x, y, xp, poly_regressor)
    print()
    
    # 3. Find Optimal Degree
    print("3. Finding Optimal Polynomial Degree:")
    optimal_degree, max_accuracy = find_optimal_degree(x, y)
    print()
    
    # 4. Final Model with Optimal Degree
    print("4. Final Model with Optimal Degree:")
    final_regressor, final_poly_features = final_polynomial_model(x, y, optimal_degree)
    print()
    
    # 5. Multiple Regression
    print("5. Multiple Regression:")
    x_multi, y_multi, multi_regressor = multiple_regression()
    plot_multiple_regression(x_multi, y_multi)
    print()
    
    print("=== DEMONSTRATION COMPLETE ===")

if __name__ == "__main__":
    main()