# K-Means Clustering Implementation
# Complete code for customer segmentation using K-Means

# Importing Libraries and Loading Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
my_file = pd.read_csv("mall_customers.csv")

# Extracting Features for Clustering
# Select columns 3 and 4 (Annual Income and Spending Score)
x = my_file.iloc[:, [3, 4]]
print("First 5 rows of extracted features:")
x.head()

# Importing K-Means and Initialising the Model
cls = KMeans(n_clusters=3, random_state=0)

# Fitting the Model and Predicting Labels
labels = cls.fit_predict(x)
print("Predicted labels:")
print(labels)

# Calculating Sum of Squared Errors (SSE) / Inertia
print("SSE/Inertia for k=3:")
print(cls.inertia_)

# Implementing the Elbow Method
sse = []
for k in range(1, 16):
    cls = KMeans(n_clusters=k, random_state=0, n_init=10)
    cls.fit_predict(x)
    sse.append(cls.inertia_)

print("SSE values for k=1 to 15:")
print(sse)

# Plotting the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 16), sse, marker='o')
plt.xlabel("Value of K (Number of Clusters)")
plt.ylabel("Sum Squared Error (SSE)")
plt.title("Elbow Method for Optimal K")
plt.grid(True, alpha=0.3)
plt.show()

# Calculating Silhouette Scores for Silhouette Analysis
silh = []
for k in range(2, 16):
    cls = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = cls.fit_predict(x)
    silh.append(silhouette_score(x, labels))

print("Silhouette scores for k=2 to 15:")
print(silh)

# Plotting Silhouette Scores
plt.figure(figsize=(10, 6))
plt.bar(range(2, 16), silh, color='skyblue', alpha=0.7)
plt.xlabel("Value of K (Number of Clusters)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal K")
plt.grid(True, alpha=0.3)
plt.show()

# Finding the Optimal 'k' from Silhouette Scores
optimal_k_index = silh.index(max(silh))
optimal_k = optimal_k_index + 2  # +2 because range started from 2
print(f"Optimal K from Silhouette Method: {optimal_k}")

# Final K-Means Model with Optimal 'k'
cls = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
labels = cls.fit_predict(x)
print("Final cluster labels:")
print(labels)

# Accessing Cluster Centres (Centroids)
print("Cluster centers (centroids):")
print(cls.cluster_centers_)

# Visualising Clustered Data and Centroids
plt.figure(figsize=(16, 9))
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(cls.cluster_centers_[:, 0], cls.cluster_centers_[:, 1], 
           s=200, c='red', marker='X', label='Centroids', edgecolors='black')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Clusters with Centroids")
plt.legend()
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# Boolean Filtering to Extract Cluster Data
cluster_0_customers = x[labels == 0]
print("Customers in Cluster 0:")
print(cluster_0_customers)

cluster_4_customers = x[labels == 4]
print("\nCustomers in Cluster 4:")
print(cluster_4_customers)

# Saving Clustered Data to a CSV File
# Add cluster labels to original dataframe
my_file['Cluster_Label'] = labels

# Save cluster 0 customers to CSV
cluster_0_full_data = my_file[my_file['Cluster_Label'] == 0]
cluster_0_full_data.to_csv("zero_customers.csv", index=False)
print("\nCluster 0 customers saved to 'zero_customers.csv'")

# Predicting Cluster for New Data Points
# Example 1: Customer with income=50k, spending score=60
new_customer = [[50, 60]]
predicted_cluster = cls.predict(new_customer)
print(f"\nNew customer (50, 60) belongs to cluster: {predicted_cluster[0]}")

# Example 2: Customer with income=15k, spending score=78
new_customer_2 = [[15, 78]]
predicted_cluster_2 = cls.predict(new_customer_2)
print(f"Another new customer (15, 78) belongs to cluster: {predicted_cluster_2[0]}")

# Example 3: Customer with income=75k, spending score=12
new_customer_3 = [[75, 12]]
predicted_cluster_3 = cls.predict(new_customer_3)
print(f"Another new customer (75, 12) belongs to cluster: {predicted_cluster_3[0]}")

# Additional Analysis: Display cluster statistics
print("\n" + "="*50)
print("CLUSTER ANALYSIS SUMMARY")
print("="*50)

for i in range(optimal_k):
    cluster_data = x[labels == i]
    print(f"\nCluster {i}:")
    print(f"  Number of customers: {len(cluster_data)}")
    print(f"  Average Annual Income: ${cluster_data.iloc[:, 0].mean():.2f}k")
    print(f"  Average Spending Score: {cluster_data.iloc[:, 1].mean():.2f}")
    print(f"  Income Range: ${cluster_data.iloc[:, 0].min():.2f}k - ${cluster_data.iloc[:, 0].max():.2f}k")
    print(f"  Spending Range: {cluster_data.iloc[:, 1].min():.2f} - {cluster_data.iloc[:, 1].max():.2f}")
