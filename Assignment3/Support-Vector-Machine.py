# Letter Recognition using Support Vector Machine (SVM)
# Complete Python implementation for letter classification

# =============================================================================
# 1. IMPORTING LIBRARIES AND LOADING DATA
# =============================================================================

import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# =============================================================================
# 2. DATA LOADING AND EXPLORATION
# =============================================================================

# Load the dataset
data = pd.read_csv("letterdata.csv")

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

# Extract target variable (output)
y = data['letter']

# Display unique classes
print("\nUnique letters:", set(y))
print("Number of unique classes:", len(set(y)))

# Extract features (input variables)
x = data.drop('letter', axis=1)

# =============================================================================
# 3. DATA SPLITTING
# =============================================================================

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=0
)

# Check class distribution in training data
c = Counter(y_train)
print("\nClass distribution in training data:")
print(c)

# =============================================================================
# 4. SVM MODEL CREATION AND TRAINING
# =============================================================================

# Option 1: Linear Kernel SVM
print("\n" + "="*50)
print("LINEAR KERNEL SVM")
print("="*50)

classifier_linear = SVC(kernel='linear', random_state=0)
classifier_linear.fit(X_train, y_train)

# Option 2: RBF Kernel SVM (Default)
print("\n" + "="*50)
print("RBF KERNEL SVM")
print("="*50)

classifier_rbf = SVC(kernel='rbf', random_state=0)
classifier_rbf.fit(X_train, y_train)

# Option 3: Polynomial Kernel SVM
print("\n" + "="*50)
print("POLYNOMIAL KERNEL SVM")
print("="*50)

classifier_poly = SVC(kernel='poly', random_state=0)
classifier_poly.fit(X_train, y_train)

# Option 4: Sigmoid Kernel SVM
print("\n" + "="*50)
print("SIGMOID KERNEL SVM")
print("="*50)

classifier_sigmoid = SVC(kernel='sigmoid', random_state=0)
classifier_sigmoid.fit(X_train, y_train)

# =============================================================================
# 5. PREDICTION AND EVALUATION
# =============================================================================

def evaluate_model(classifier, kernel_name, X_test, y_test):
    """
    Evaluate a trained SVM classifier and display results
    """
    print(f"\n{kernel_name} Kernel Results:")
    print("-" * 40)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Generate confusion matrix
    mat = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(mat)
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix heatmap
    plt.figure(figsize=(16, 9))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title(f'Confusion Matrix - {kernel_name} Kernel')
    plt.show()
    
    return y_pred, accuracy

# Evaluate all models
results = {}

# Linear Kernel
y_pred_linear, acc_linear = evaluate_model(
    classifier_linear, "Linear", X_test, y_test
)
results['Linear'] = acc_linear

# RBF Kernel
y_pred_rbf, acc_rbf = evaluate_model(
    classifier_rbf, "RBF", X_test, y_test
)
results['RBF'] = acc_rbf

# Polynomial Kernel
y_pred_poly, acc_poly = evaluate_model(
    classifier_poly, "Polynomial", X_test, y_test
)
results['Polynomial'] = acc_poly

# Sigmoid Kernel
y_pred_sigmoid, acc_sigmoid = evaluate_model(
    classifier_sigmoid, "Sigmoid", X_test, y_test
)
results['Sigmoid'] = acc_sigmoid

# =============================================================================
# 6. SINGLE ENTRY PREDICTION
# =============================================================================

print("\n" + "="*50)
print("SINGLE ENTRY PREDICTION")
print("="*50)

# Select a random entry from training data for prediction
# Note: Using index 10567 as example (adjust based on your dataset size)
try:
    single_entry = X_train.iloc[10567].values.reshape(1, -1)
    actual_letter = y_train.iloc[10567]
    
    print(f"Actual letter: {actual_letter}")
    
    # Predict using different kernels
    pred_linear = classifier_linear.predict(single_entry)[0]
    pred_rbf = classifier_rbf.predict(single_entry)[0]
    pred_poly = classifier_poly.predict(single_entry)[0]
    pred_sigmoid = classifier_sigmoid.predict(single_entry)[0]
    
    print(f"Linear kernel prediction: {pred_linear}")
    print(f"RBF kernel prediction: {pred_rbf}")
    print(f"Polynomial kernel prediction: {pred_poly}")
    print(f"Sigmoid kernel prediction: {pred_sigmoid}")
    
except IndexError:
    print("Index 10567 not available. Using first entry instead.")
    single_entry = X_train.iloc[0].values.reshape(1, -1)
    actual_letter = y_train.iloc[0]
    
    print(f"Actual letter: {actual_letter}")
    
    # Predict using different kernels
    pred_linear = classifier_linear.predict(single_entry)[0]
    pred_rbf = classifier_rbf.predict(single_entry)[0]
    pred_poly = classifier_poly.predict(single_entry)[0]
    pred_sigmoid = classifier_sigmoid.predict(single_entry)[0]
    
    print(f"Linear kernel prediction: {pred_linear}")
    print(f"RBF kernel prediction: {pred_rbf}")
    print(f"Polynomial kernel prediction: {pred_poly}")
    print(f"Sigmoid kernel prediction: {pred_sigmoid}")

# =============================================================================
# 7. RESULTS SUMMARY
# =============================================================================

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)

print("Model Accuracies:")
for kernel, accuracy in results.items():
    print(f"{kernel:12}: {accuracy:.4f}")

best_kernel = max(results, key=results.get)
print(f"\nBest performing kernel: {best_kernel} ({results[best_kernel]:.4f})")

# =============================================================================
# 8. OPTIONAL: SUPPORT VECTOR REGRESSION (SVR)
# =============================================================================

# Note: SVR is used for regression problems, not classification
# Uncomment the following lines if you need to use SVR for regression tasks

# from sklearn.svm import SVR
# svr_classifier = SVR(kernel='rbf')
# # SVR would be used for continuous target variables, not discrete letters