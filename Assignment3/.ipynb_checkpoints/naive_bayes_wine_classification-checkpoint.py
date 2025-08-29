# Naive Bayes Wine Classification Implementation
# Demonstrates both Gaussian and Multinomial Naive Bayes classifiers

# ===== 1. DATA LOADING AND PREPARATION =====

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the wine dataset (assuming you have a wine dataset CSV file)
# data = pd.read_csv('wine_dataset.csv')  # Replace with your dataset path

# Separate features (X) and target (y)
# The 'class' column is the output variable, remaining 13 columns are input features
X = data.drop('class', axis=1)  # All 13 feature columns
y = data['class']  # Target variable with classes 1, 2, 3

# Display basic information about the data
print("Dataset shape:", data.shape)  # Should be (178, 14)
print("Features shape:", X.shape)    # Should be (178, 13)
print("Unique classes:", y.unique()) # Should show [1, 2, 3]
print("\nFirst 5 rows of features:")
print(X.head())

# Split data into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"\nTraining set shape: {X_train.shape}")  # Should be (133, 13)
print(f"Testing set shape: {X_test.shape}")     # Should be (45, 13)

# ===== 2. GAUSSIAN NAIVE BAYES CLASSIFIER =====

print("\n" + "="*50)
print("GAUSSIAN NAIVE BAYES CLASSIFIER")
print("="*50)

# Create and train Gaussian Naive Bayes classifier
gaussian_classifier = GaussianNB()
gaussian_classifier.fit(X_train, y_train)
print("Gaussian Naive Bayes model trained successfully!")

# Make predictions on test data
y_pred_gaussian = gaussian_classifier.predict(X_test)
print(f"Predictions made for {len(y_pred_gaussian)} test samples")

# Evaluate Gaussian Naive Bayes performance
print("\n--- Gaussian Naive Bayes Evaluation ---")

# Confusion Matrix
print("Confusion Matrix:")
cm_gaussian = confusion_matrix(y_test, y_pred_gaussian)
print(cm_gaussian)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gaussian))

# Accuracy Score
accuracy_gaussian = accuracy_score(y_test, y_pred_gaussian)
print(f"Accuracy: {accuracy_gaussian:.4f} ({accuracy_gaussian*100:.2f}%)")

# Predict probabilities for test data
print("\n--- Probability Predictions (first 5 samples) ---")
probabilities_gaussian = gaussian_classifier.predict_proba(X_test)
for i in range(min(5, len(probabilities_gaussian))):
    print(f"Sample {i+1}: {probabilities_gaussian[i]}")

# Single entry prediction example
print("\n--- Single Wine Prediction Example ---")
# Example wine features (all 13 features required)
new_wine_features = [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 360]
new_wine_df = pd.DataFrame([new_wine_features], columns=X.columns)

single_prediction_gaussian = gaussian_classifier.predict(new_wine_df)
single_probability_gaussian = gaussian_classifier.predict_proba(new_wine_df)

print(f"New wine predicted class: {single_prediction_gaussian[0]}")
print(f"Class probabilities: {single_probability_gaussian[0]}")

# ===== 3. MULTINOMIAL NAIVE BAYES CLASSIFIER =====

print("\n" + "="*50)
print("MULTINOMIAL NAIVE BAYES CLASSIFIER")
print("="*50)

# Create and train Multinomial Naive Bayes classifier
multinomial_classifier = MultinomialNB()
multinomial_classifier.fit(X_train, y_train)
print("Multinomial Naive Bayes model trained successfully!")

# Make predictions on test data
y_pred_multinomial = multinomial_classifier.predict(X_test)
print(f"Predictions made for {len(y_pred_multinomial)} test samples")

# Evaluate Multinomial Naive Bayes performance
print("\n--- Multinomial Naive Bayes Evaluation ---")

# Confusion Matrix
print("Confusion Matrix:")
cm_multinomial = confusion_matrix(y_test, y_pred_multinomial)
print(cm_multinomial)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_multinomial))

# Accuracy Score
accuracy_multinomial = accuracy_score(y_test, y_pred_multinomial)
print(f"Accuracy: {accuracy_multinomial:.4f} ({accuracy_multinomial*100:.2f}%)")

# Single entry prediction with Multinomial Naive Bayes
print("\n--- Single Wine Prediction with Multinomial NB ---")
single_prediction_multinomial = multinomial_classifier.predict(new_wine_df)
single_probability_multinomial = multinomial_classifier.predict_proba(new_wine_df)

print(f"New wine predicted class: {single_prediction_multinomial[0]}")
print(f"Class probabilities: {single_probability_multinomial[0]}")

# ===== 4. MODEL COMPARISON =====

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

print(f"Gaussian Naive Bayes Accuracy: {accuracy_gaussian:.4f} ({accuracy_gaussian*100:.2f}%)")
print(f"Multinomial Naive Bayes Accuracy: {accuracy_multinomial:.4f} ({accuracy_multinomial*100:.2f}%)")

print("\nConfusion Matrix Comparison:")
print("Gaussian NB:")
print(cm_gaussian)
print("\nMultinomial NB:")
print(cm_multinomial)

# Compare predictions for the single wine example
print(f"\nSingle Wine Prediction Comparison:")
print(f"Gaussian NB predicted: Class {single_prediction_gaussian[0]}")
print(f"Multinomial NB predicted: Class {single_prediction_multinomial[0]}")

if single_prediction_gaussian[0] == single_prediction_multinomial[0]:
    print("Both models agree on the prediction!")
else:
    print("Models disagree on the prediction.")