# Logistic Regression Implementation with scikit-learn
# Complete formatted code from YouTube video transcript

# 1. Initial Setup and Data Loading
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("Social_Network_Ads.csv")

# Display the shape of the dataset (rows, columns)
print("Dataset Shape:", data.shape)

# Check for missing values
print("Missing Values:\n", data.isnull().sum())

# Display column names
print("Column Names:", data.columns)

# Select input (X) and output (y) variables
# Age and Estimated Salary (columns 2 and 3) are chosen as input
# Purchased (last column) is chosen as the output
X = data.iloc[:, 2:4].values  # Age and Estimated Salary
y = data.iloc[:, -1].values   # Purchased

# Display unique values in the output variable to confirm binary classification
print("Unique Output Values:", np.unique(y))

# Display descriptive statistics for the initial data
print("Descriptive Statistics for X:\n", pd.DataFrame(X, columns=['Age', 'EstimatedSalary']).describe())

# 2. Splitting Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# Split the data with a 75% training size and 25% testing size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Verify the lengths of the split datasets
print("Training data length (X_train):", len(X_train))  # Expected: 300
print("Testing data length (X_test):", len(X_test))     # Expected: 100

# 3. Training Logistic Regression Model (Initial Attempt - without scaling)
from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression classifier object
classifier = LogisticRegression()

# Train the model using the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred_unscaled = classifier.predict(X_test)

# Print predicted outputs
print("Predicted outputs (unscaled data):\n", y_pred_unscaled)

# Print actual outputs for comparison
print("Actual outputs (y_test):\n", y_test)

# Import confusion_matrix for evaluation
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm_unscaled = confusion_matrix(y_test, y_pred_unscaled)
print("Confusion Matrix (unscaled data):\n", cm_unscaled)

# 4. Feature Scaling
from sklearn.preprocessing import MinMaxScaler
# Alternative: from sklearn.preprocessing import StandardScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()
# Alternative: scaler = StandardScaler()

# Fit the scaler to the entire input data (X) and transform it
X_scaled = scaler.fit_transform(X)

# Display descriptive statistics of the scaled data
print("Descriptive Statistics for Scaled X:\n", pd.DataFrame(X_scaled, columns=['Age_Scaled', 'EstimatedSalary_Scaled']).describe())

# 5. Re-training and Evaluating Model with Scaled Data
# Re-split the scaled data into training and testing sets
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

# Create a new Logistic Regression classifier object
classifier_scaled = LogisticRegression()

# Train the model using the scaled training data
classifier_scaled.fit(X_train_scaled, y_train_scaled)

# Make predictions on the scaled test data
y_pred_scaled = classifier_scaled.predict(X_test_scaled)

# Create a DataFrame to compare actual vs. predicted values
result_df = pd.DataFrame({'Actual': y_test_scaled, 'Predicted': y_pred_scaled})
print("Actual vs. Predicted (scaled data):\n", result_df.head(20))

# Import evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Calculate and print the confusion matrix for the scaled model
cm_scaled = confusion_matrix(y_test_scaled, y_pred_scaled)
print("Confusion Matrix (scaled data):\n", cm_scaled)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test_scaled, y_pred_scaled)
print("Accuracy Score (scaled data):", accuracy)  # Expected: 0.89 or 89%

# Print the classification report
print("Classification Report (scaled data):\n", classification_report(y_test_scaled, y_pred_scaled))

# 6. Single Entry Prediction
# Example 1: High age and high salary (expected to purchase)
age1 = 56
salary1 = 120000

# Create a NumPy array for the new data point
new_data1 = np.array([[age1, salary1]])

# Scale the new data point using the previously fitted scaler
new_data1_scaled = scaler.transform(new_data1)

# Make a prediction using the scaled model
prediction1 = classifier_scaled.predict(new_data1_scaled)

# Interpret and print the prediction
if prediction1 == 1:
    print(f"For Age {age1}, Salary {salary1}: Will purchase the item.")
else:
    print(f"For Age {age1}, Salary {salary1}: Will not purchase the item.")

# Example 2: Low age and low salary (expected not to purchase)
age2 = 27
salary2 = 30000

new_data2 = np.array([[age2, salary2]])
new_data2_scaled = scaler.transform(new_data2)
prediction2 = classifier_scaled.predict(new_data2_scaled)

if prediction2 == 1:
    print(f"For Age {age2}, Salary {salary2}: Will purchase the item.")
else:
    print(f"For Age {age2}, Salary {salary2}: Will not purchase the item.")

# 7. Predicting Probabilities
# Predict probabilities for the scaled test data
y_pred_proba = classifier_scaled.predict_proba(X_test_scaled)
print("Predicted Probabilities (first 10 entries):\n", y_pred_proba[:10])

# 8. ROC AUC Curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Get the probabilities for the positive class (class 1)
y_pred_proba_class1 = classifier_scaled.predict_proba(X_test_scaled)[:, 1]

# Calculate False Positive Rate (fpr), True Positive Rate (tpr), and thresholds
fpr, tpr, thresholds = roc_curve(y_test_scaled, y_pred_proba_class1)

# Calculate the Area Under the Curve (AUC) score
auc_score = roc_auc_score(y_test_scaled, y_pred_proba_class1)
print(f"ROC AUC Score: {auc_score:.4f}")

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 9. Manual Calculation of Performance Metrics from Confusion Matrix
# Extract values from confusion matrix
# cm_scaled structure: [[TN, FP], [FN, TP]]
tn, fp, fn, tp = cm_scaled.ravel()

# Total observations in the test set
total_observations = np.sum(cm_scaled)

# Accuracy: (True Positives + True Negatives) / Total Observations
accuracy_manual = (tp + tn) / total_observations
print(f"Manual Accuracy: {accuracy_manual:.4f}")

# Sensitivity (Recall for Positive Class): True Positive / (True Positive + False Negative)
sensitivity = tp / (tp + fn)
print(f"Manual Sensitivity (Recall for Class 1): {sensitivity:.4f}")

# Specificity: True Negative / (True Negative + False Positive)
specificity = tn / (tn + fp)
print(f"Manual Specificity: {specificity:.4f}")

# False Positive Rate (FPR): False Positive / (True Negative + False Positive)
false_positive_rate = fp / (tn + fp)
print(f"Manual False Positive Rate: {false_positive_rate:.4f}")
