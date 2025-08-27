# Data Preprocessing for Machine Learning - Python Code Reference

# =============================================================================
# 1. IMPORT REQUIRED LIBRARIES
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer, LabelEncoder

# =============================================================================
# 2. READING AND INITIAL DATA EXPLORATION
# =============================================================================

# Read CSV file
data = pd.read_csv("student.csv")

# Get dataset dimensions (rows, columns)
print("Dataset shape:", data.shape)

# View first 5 rows
print("\nFirst 5 rows:")
print(data.head())

# Get statistical summary of numerical columns
print("\nStatistical summary:")
print(data.describe())

# =============================================================================
# 3. HANDLING MISSING VALUES
# =============================================================================

# Identifying Missing Values
print("\nNon-null values in each column:")
print(data.count())

print("\nMissing values in each column:")
print(data.isnull().sum())

# Basic Statistics for Missing Value Handling
print("\nMean of marks column:", data['marks'].mean())
print("Maximum age:", data['age'].max())
print("Mode of class column:", data['class'].mode())

# Missing Value Treatment Methods

# Method 1: Remove rows with any missing values
data_cleaned = data.dropna()

# Method 2: Fill all missing values with a constant (e.g., 0)
data_filled = data.fillna(0)

# Method 3: Forward fill (use previous valid observation)
data_filled = data.fillna(method='pad')

# Method 4: Backward fill (use next valid observation)
data_filled = data.fillna(method='backfill')

# Method 5: Fill numerical column with median
data['marks'] = data['marks'].fillna(data['marks'].median())

# Method 6: Fill numerical column with mean
data['marks'] = data['marks'].fillna(data['marks'].mean())

# Method 7: Fill categorical column with mode
data['class'] = data['class'].fillna(data['class'].mode()[0])

# =============================================================================
# 4. DATA SPLITTING
# =============================================================================

# Assuming X and y are defined (features and target)
# X = data.drop('target_column', axis=1)
# y = data['target_column']

# Split data into training and testing sets
# 75% train, 25% test with fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=42
)

# =============================================================================
# 5. FEATURE SCALING AND NORMALIZATION
# =============================================================================

# Standard Scaling (Standardization)
# Creates features with mean=0 and std=1
scaler_standard = StandardScaler()
X_standard_scaled = scaler_standard.fit_transform(X)

# Min-Max Scaling (Normalization)
# Scales features to a range between 0 and 1
scaler_minmax = MinMaxScaler()
X_minmax_scaled = scaler_minmax.fit_transform(X)

# =============================================================================
# 6. BINARIZATION
# =============================================================================

# Converting Continuous Features to Binary
# Values above threshold become 1, below become 0
binarizer = Binarizer(threshold=30)
X_binned = binarizer.fit_transform(X)

# =============================================================================
# 7. ENCODING CATEGORICAL VARIABLES
# =============================================================================

# Label Encoding
# Converts categorical values to numerical labels
le = LabelEncoder()
data['Name'] = le.fit_transform(data['Name'])

# One-Hot Encoding
# Creates binary columns for each category
data_encoded = pd.get_dummies(data)

# =============================================================================
# 8. COMPLETE DATA PREPROCESSING WORKFLOW
# =============================================================================

def preprocess_data(filepath, target_column):
    """
    Complete data preprocessing pipeline
    
    Parameters:
    filepath (str): Path to the CSV file
    target_column (str): Name of the target column
    
    Returns:
    tuple: Preprocessed training and testing data
    """
    
    # 1. Load data
    data = pd.read_csv(filepath)
    print(f"Dataset loaded with shape: {data.shape}")
    
    # 2. Explore data
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nDataset info:")
    print(data.describe())
    print("\nMissing values:")
    print(data.isnull().sum())
    
    # 3. Handle missing values
    # For numerical columns: fill with mean
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        if col != target_column:
            data[col] = data[col].fillna(data[col].mean())
    
    # For categorical columns: fill with mode
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != target_column:
            data[col] = data[col].fillna(data[col].mode()[0])
    
    # 4. Encode categorical variables
    le_dict = {}
    for col in categorical_cols:
        if col != target_column:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            le_dict[col] = le
    
    # 5. Split features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # 6. Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # 7. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nPreprocessing complete!")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le_dict

# =============================================================================
# 9. EXAMPLE USAGE
# =============================================================================

# Example usage of the preprocessing pipeline
if __name__ == "__main__":
    # Replace with your actual file path and target column name
    # X_train, X_test, y_train, y_test, scaler, encoders = preprocess_data("your_dataset.csv", "target_column")
    
    # For individual operations, you can use the functions above
    # Make sure to always fit on training data and transform both train and test
    pass

# =============================================================================
# KEY NOTES AND BEST PRACTICES
# =============================================================================

"""
Important Notes:
1. Always fit scalers on training data only, then transform both training and test sets
2. Use random_state parameter for reproducible results
3. Choose appropriate missing value handling strategies based on your data
4. Consider the nature of your categorical variables when choosing encoding methods
5. Feature scaling is crucial for algorithms sensitive to feature magnitudes (SVM, Neural Networks, etc.)
6. Save your preprocessing objects (scalers, encoders) for use on new data
7. Handle missing values before encoding categorical variables
8. Be careful with data leakage - don't use information from test set during preprocessing
"""