# Hierarchical Clustering in Python
# Complete code examples with proper formatting

# ==============================================================================
# REQUIRED PACKAGES
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# ==============================================================================
# 1. DENDROGRAM DEMONSTRATION WITH SAMPLE DATA
# ==============================================================================

# Create sample 2D data points
x = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], 
              [10, 0], [5, 6], [5, 8], [5, 4], [3, 3]])

# Plot scatter plot of sample data
plt.figure(figsize=(8, 6))
plt.scatter(x[:, 0], x[:, 1], s=100, alpha=0.7)
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Sample Data Points')
plt.grid(True, alpha=0.3)
plt.show()

# Generate linkage matrix for dendrogram
linked = linkage(x, method='single')  # Using 'single' linkage criteria

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(
    linked,
    orientation='top',
    labels=range(1, 11),  # Labels for the 10 data points
    distance_sort='descending',
    show_leaf_counts=True  # Show counts of original observations in each leaf
)
plt.title('Dendrogram for Sample Data')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.show()

# ==============================================================================
# 2. HIERARCHICAL CLUSTERING ON CUSTOMER DATA
# ==============================================================================

# Load customer dataset
data = pd.read_csv('Mall_Customers.csv')

# Select relevant columns: Annual Income and Spending Score
# Assuming columns 3 and 4 are the target features
x = data.iloc[:, [3, 4]].values

# Generate and plot dendrogram for customer data
plt.figure(figsize=(16, 9))
plt.title('Customer Dendrogram')
dendrogram(linkage(x, method='ward'))
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.show()

# Apply Agglomerative Clustering
cls = AgglomerativeClustering(
    n_clusters=5, 
    affinity='euclidean', 
    linkage='ward'
)

# Fit model and predict cluster labels
labels = cls.fit_predict(x)
print("Cluster labels:", labels)
print("Number of points in each cluster:", np.bincount(labels))

# Visualize the clusters
plt.figure(figsize=(16, 9))
plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='rainbow', s=50, alpha=0.7)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Hierarchical Clusters of Customers')
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# Save data from a specific cluster (e.g., cluster 4)
cluster_4_data = data[labels == 4]
cluster_4_data.to_csv('cluster_4_data.csv', index=False)
print(f"Saved {len(cluster_4_data)} data points from cluster 4")

# ==============================================================================
# 3. GAUSSIAN MIXTURE CLUSTERING (FOR COMPARISON)
# ==============================================================================

# Create Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, random_state=42)

# Fit model and predict labels
gmm_labels = gmm.fit_predict(x)
print("GMM cluster labels:", gmm_labels)
print("Number of points in each GMM cluster:", np.bincount(gmm_labels))

# Visualize Gaussian Mixture clusters
plt.figure(figsize=(16, 9))
plt.scatter(x[:, 0], x[:, 1], c=gmm_labels, cmap='rainbow', s=50, alpha=0.7)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Gaussian Mixture Clusters of Customers')
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# ==============================================================================
# 4. COMPARISON OF DIFFERENT LINKAGE METHODS
# ==============================================================================

# Compare different linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, method in enumerate(linkage_methods):
    # Generate linkage matrix
    if method == 'ward':
        # Ward method requires squared Euclidean distance
        linked = linkage(x, method=method)
    else:
        linked = linkage(x, method=method, metric='euclidean')
    
    # Plot dendrogram
    axes[i].set_title(f'Dendrogram - {method.title()} Linkage')
    dendrogram(linked, ax=axes[i], leaf_rotation=90, leaf_font_size=8)
    axes[i].set_xlabel('Sample Index or (Cluster Size)')
    axes[i].set_ylabel('Distance')

plt.tight_layout()
plt.show()

# ==============================================================================
# 5. OPTIMAL NUMBER OF CLUSTERS ANALYSIS
# ==============================================================================

# Function to calculate within-cluster sum of squares
def calculate_wcss(data, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 1):
        cls = AgglomerativeClustering(n_clusters=i, linkage='ward')
        labels = cls.fit_predict(data)
        
        # Calculate WCSS
        wcss_value = 0
        for cluster in range(i):
            cluster_points = data[labels == cluster]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                wcss_value += np.sum((cluster_points - centroid) ** 2)
        wcss.append(wcss_value)
    
    return wcss

# Calculate WCSS for different numbers of clusters
wcss = calculate_wcss(x, max_clusters=10)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linewidth=2, markersize=8)
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True, alpha=0.3)
plt.show()

print("WCSS values:", wcss)