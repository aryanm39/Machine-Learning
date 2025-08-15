# K-Nearest Neighbors Implementation
# Complete code for both classification (Iris) and regression (Turnout) tasks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =============================================================================
# 1. IRIS DATASET - CLASSIFICATION
# =============================================================================

# Method 1: Loading Iris Dataset from URL
def load_iris_from_url():
    """Load Iris dataset from UCI repository"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    dataset = pd.read_csv(url, names=column_names)
    return dataset

# Method 2: Loading Iris Dataset from sklearn (Recommended)
def load_iris_from_sklearn():
    """Load Iris dataset from sklearn and convert to DataFrame"""
    # Load the Iris dataset
    x_bunch = load_iris()
    
    # Convert to DataFrame with proper column names
    combined_data = np.c_[x_bunch.data, x_bunch.target]
    df_columns = x_bunch.feature_names + ['target']
    iris_df = pd.DataFrame(data=combined_data, columns=df_columns)
    
    # Replace numeric targets with class names
    t1 = pd.DataFrame(x_bunch.target)
    t1.replace(0, x_bunch.target_names[0], inplace=True)
    t1.replace(1, x_bunch.target_names[1], inplace=True)
    t1.replace(2, x_bunch.target_names[2], inplace=True)
    
    # Create final DataFrame with named targets
    iris_df_named = pd.DataFrame(np.c_[x_bunch.data, t1], 
                                columns=x_bunch.feature_names + ['target'])
    
    return iris_df_named, x_bunch

# Data Preprocessing for Classification
def preprocess_iris_data(iris_df):
    """Preprocess Iris data for KNN classification"""
    # Separate features and target
    X = iris_df.drop('target', axis=1)
    y = iris_df['target']
    
    # Min-Max Scaling (normalize features to 0-1 range)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=0
    )
    
    return X_train, X_test, y_train, y_test, scaler

# KNN Classification Implementation
def train_knn_classifier(X_train, y_train, n_neighbors=5):
    """Train KNN classifier"""
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(X_train, y_train)
    return knn_classifier

# Model Evaluation
def evaluate_classifier(model, X_test, y_test):
    """Evaluate KNN classifier performance"""
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy Score: {accuracy*100:.2f}%")
    
    return y_pred, accuracy

# Find Optimal K Value
def find_optimal_k(X_train, X_test, y_train, y_test, max_k=113):
    """Find optimal K value by plotting error rates"""
    error_rate = []
    
    # Calculate error rates for different K values
    for i in range(1, max_k):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
    
    # Plot error rate vs K value
    plt.figure(figsize=(16, 9))
    plt.plot(range(1, len(error_rate) + 1), error_rate,
             color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.xlabel('Value of K')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs. K Value for Iris Dataset')
    plt.show()
    
    # Find K with minimum error
    optimal_k = np.argmin(error_rate) + 1
    print(f"Optimal K value: {optimal_k}")
    print(f"Minimum error rate: {min(error_rate):.4f}")
    
    return optimal_k, error_rate

# Single Entry Prediction
def predict_single_entry(model, scaler, new_data):
    """Predict class for a single new data point"""
    # Scale the new data using the same scaler
    scaled_new_data = scaler.transform(new_data)
    
    # Make prediction
    prediction = model.predict(scaled_new_data)
    return prediction

# =============================================================================
# 2. TURNOUT DATASET - REGRESSION
# =============================================================================

def load_turnout_data(file_path="turnout.csv"):
    """Load turnout dataset for regression"""
    turnout_df = pd.read_csv(file_path)
    return turnout_df

def preprocess_turnout_data(turnout_df):
    """Preprocess turnout data for KNN regression"""
    # Separate features and target
    X = turnout_df.drop('educate', axis=1)  # 'educate' is the continuous target
    y = turnout_df['educate']
    
    # Label encode categorical variables (e.g., 'race' column)
    le = LabelEncoder()
    X['race'] = le.fit_transform(X['race'])
    
    # Standard Scaling for regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=0
    )
    
    return X_train, X_test, y_train, y_test, scaler, le

def train_knn_regressor(X_train, y_train, n_neighbors=10):
    """Train KNN regressor"""
    knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_regressor.fit(X_train, y_train)
    return knn_regressor

def evaluate_regressor(model, X_test, y_test):
    """Evaluate KNN regressor performance"""
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    
    return y_pred, mae, mse

# =============================================================================
# 3. MAIN EXECUTION FUNCTIONS
# =============================================================================

def run_iris_classification():
    """Complete workflow for Iris classification"""
    print("=== IRIS DATASET CLASSIFICATION ===")
    
    # Load data
    iris_df, x_bunch = load_iris_from_sklearn()
    print(f"Dataset shape: {iris_df.shape}")
    print(f"Classes: {x_bunch.target_names}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_iris_data(iris_df)
    
    # Train initial model
    print("\n--- Training with K=5 ---")
    knn_model = train_knn_classifier(X_train, y_train, n_neighbors=5)
    evaluate_classifier(knn_model, X_test, y_test)
    
    # Find optimal K
    print("\n--- Finding Optimal K ---")
    optimal_k, error_rates = find_optimal_k(X_train, X_test, y_train, y_test)
    
    # Train with optimal K
    print(f"\n--- Training with Optimal K={optimal_k} ---")
    knn_optimal = train_knn_classifier(X_train, y_train, n_neighbors=optimal_k)
    evaluate_classifier(knn_optimal, X_test, y_test)
    
    # Example predictions
    print("\n--- Single Entry Predictions ---")
    new_samples = [
        np.array([[5.0, 3.2, 1.2, 0.2]]),  # Expected: setosa
        np.array([[6.0, 2.2, 5.0, 1.5]]),  # Expected: versicolor
        np.array([[7.0, 3.2, 6.0, 2.0]])   # Expected: virginica
    ]
    
    for i, sample in enumerate(new_samples, 1):
        prediction = predict_single_entry(knn_optimal, scaler, sample)
        print(f"Sample {i}: {sample.flatten()} -> Predicted: {prediction[0]}")

def run_turnout_regression():
    """Complete workflow for Turnout regression"""
    print("\n=== TURNOUT DATASET REGRESSION ===")
    
    try:
        # Load data
        turnout_df = load_turnout_data()
        print(f"Dataset shape: {turnout_df.shape}")
        
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler, le = preprocess_turnout_data(turnout_df)
        
        # Train regressor
        print("\n--- Training KNN Regressor ---")
        knn_regressor = train_knn_regressor(X_train, y_train, n_neighbors=10)
        
        # Evaluate regressor
        evaluate_regressor(knn_regressor, X_test, y_test)
        
    except FileNotFoundError:
        print("turnout.csv file not found. Please ensure the file is available.")

# =============================================================================
# 4. EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Run Iris classification example
    run_iris_classification()
    
    # Run Turnout regression example (if data available)
    run_turnout_regression()
