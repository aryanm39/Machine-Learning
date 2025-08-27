# Decision Tree Implementation - Classification and Regression
# This code demonstrates both classification and regression using Decision Trees

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report,
                           mean_absolute_error, mean_squared_error)
from io import StringIO
from IPython.display import Image
import pydotplus

# ===========================
# DECISION TREE CLASSIFICATION
# ===========================

def decision_tree_classification():
    """
    Implement Decision Tree Classification for banknote authentication
    """
    print("=== DECISION TREE CLASSIFICATION ===\n")
    
    # 1. Loading and Exploring Data
    print("1. Loading and exploring data...")
    data = pd.read_csv("banknotes.csv")
    
    print(f"Dataset shape: {data.shape}")
    print(f"Missing values:\n{data.isnull().sum()}")
    
    # 2. Preparing Features and Target
    print("\n2. Preparing features and target variables...")
    X = data.drop('class', axis=1)  # Input variables
    y = data['class']               # Output variable
    
    print("Feature variables (X):")
    print(X.head())
    print(f"\nUnique classes: {np.unique(y)}")
    print(f"Alternative method: {set(y)}")
    
    # 3. Splitting Data
    print("\n3. Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print(f"Training set size: {len(X_train)}")
    
    # 4. Building and Training the Model
    print("\n4. Building and training Decision Tree Classifier...")
    classifier = DecisionTreeClassifier(random_state=0)
    classifier.fit(X_train, y_train)
    print("Model trained successfully!")
    
    # 5. Making Predictions
    print("\n5. Making predictions...")
    y_pred = classifier.predict(X_test)
    
    # 6. Model Evaluation
    print("\n6. Evaluating the model...")
    
    # Actual vs Predicted comparison
    df_results = pd.DataFrame({
        'Actual Values': y_test, 
        'Predicted Values': y_pred
    })
    print("Sample predictions:")
    print(df_results.head())
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Accuracy Score
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
    
    # Classification Report
    report = classification_report(y_test, y_pred)
    print(f"\nClassification Report:\n{report}")
    
    # 7. Visualizing Decision Tree
    print("\n7. Generating decision tree visualization...")
    try:
        dot_data = StringIO()
        export_graphviz(classifier, out_file=dot_data,
                       filled=True, rounded=True,
                       special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('tree.png')
        print("Decision tree saved as 'tree.png'")
        # Image(graph.create_png())  # For Jupyter notebook display
    except Exception as e:
        print(f"Visualization error: {e}")
    
    # 8. Single Prediction Example
    print("\n8. Single prediction example...")
    variance = 0.6464
    skewness = 4.0
    kurtosis = 1.0
    entropy = -0.5
    
    new_note_features = np.array([[variance, skewness, kurtosis, entropy]])
    prediction = classifier.predict(new_note_features)
    
    if prediction == 1:
        print("Note Accepted (Authentic)")
    else:
        print("Note Rejected (Fake)")
    
    return classifier


# ===========================
# DECISION TREE REGRESSION
# ===========================

def decision_tree_regression():
    """
    Implement Decision Tree Regression for petrol consumption prediction
    """
    print("\n\n=== DECISION TREE REGRESSION ===\n")
    
    # 1. Loading and Exploring Data
    print("1. Loading and exploring data...")
    data_reg = pd.read_csv("petrol_consumption.csv")
    print(f"Dataset shape: {data_reg.shape}")
    
    # 2. Preparing Features and Target
    print("\n2. Preparing features and target variables...")
    X_reg = data_reg.drop('Petrol_Consumption', axis=1)  # Input variables
    y_reg = data_reg['Petrol_Consumption']               # Output variable
    
    print("Feature variables (X_reg):")
    print(X_reg.head())
    
    # 3. Splitting Data
    print("\n3. Splitting data into training and testing sets...")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, random_state=0
    )
    print(f"Training set size: {len(X_train_reg)}")
    
    # 4. Building and Training the Model
    print("\n4. Building and training Decision Tree Regressor...")
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train_reg, y_train_reg)
    print("Model trained successfully!")
    
    # 5. Making Predictions
    print("\n5. Making predictions...")
    y_pred_reg = regressor.predict(X_test_reg)
    
    # 6. Model Evaluation
    print("\n6. Evaluating the model...")
    
    # Actual vs Predicted comparison
    df_reg_results = pd.DataFrame({
        'Actual Values': y_test_reg, 
        'Predicted Values': y_pred_reg
    })
    print("Prediction results:")
    print(df_reg_results)
    
    # Regression Metrics
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    
    print(f"\nMean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # Custom Accuracy Calculation
    accuracy_reg = 100 - (np.mean(np.abs(y_test_reg - y_pred_reg) / y_test_reg) * 100)
    print(f"Accuracy: {accuracy_reg:.2f}%")
    
    # 7. Single Prediction Example
    print("\n7. Single prediction example...")
    petrol_tax = 7
    avg_income = 9865
    paved_highways = 586
    driver_license = 0.58
    
    new_state_features = np.array([[petrol_tax, avg_income, paved_highways, driver_license]])
    predicted_consumption = regressor.predict(new_state_features)
    print(f"Predicted petrol consumption: {predicted_consumption[0]:.2f}")
    
    return regressor


# ===========================
# MAIN EXECUTION
# ===========================

if __name__ == "__main__":
    print("Decision Tree Implementation Demo")
    print("=" * 50)
    
    # Run Classification Example
    try:
        classifier_model = decision_tree_classification()
    except FileNotFoundError:
        print("Note: banknotes.csv file not found. Please ensure the file exists.")
    except Exception as e:
        print(f"Classification error: {e}")
    
    # Run Regression Example
    try:
        regressor_model = decision_tree_regression()
    except FileNotFoundError:
        print("Note: petrol_consumption.csv file not found. Please ensure the file exists.")
    except Exception as e:
        print(f"Regression error: {e}")
    
    print("\n" + "=" * 50)
    print("Demo completed!")


# ===========================
# UTILITY FUNCTIONS
# ===========================

def predict_banknote(classifier, variance, skewness, kurtosis, entropy):
    """
    Utility function to predict if a banknote is authentic or fake
    
    Args:
        classifier: Trained DecisionTreeClassifier
        variance: Variance of banknote image
        skewness: Skewness of banknote image
        kurtosis: Kurtosis of banknote image
        entropy: Entropy of banknote image
    
    Returns:
        str: "Authentic" or "Fake"
    """
    features = np.array([[variance, skewness, kurtosis, entropy]])
    prediction = classifier.predict(features)
    return "Authentic" if prediction == 1 else "Fake"


def predict_petrol_consumption(regressor, petrol_tax, avg_income, paved_highways, driver_license):
    """
    Utility function to predict petrol consumption
    
    Args:
        regressor: Trained DecisionTreeRegressor
        petrol_tax: Petrol tax rate
        avg_income: Average income
        paved_highways: Number of paved highways
        driver_license: Driver license ratio
    
    Returns:
        float: Predicted petrol consumption
    """
    features = np.array([[petrol_tax, avg_income, paved_highways, driver_license]])
    return regressor.predict(features)[0]