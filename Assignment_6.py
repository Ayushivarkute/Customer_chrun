import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Load your dataset (replace the path with your actual file path)
dataset = pd.read_csv("C:/Users/ayush/Downloads/data.csv")

# Extract the feature matrix (X should be 2D for clustering)
X = dataset[['Height', 'Weight']]  # Adjust columns accordingly

# Range of clusters (k) to try
min_clusters = 2
max_clusters = 10

# Initialize list to store within-cluster sum of squares (WCSS)
wcss = []

# Iterate over different values of k
for k in range(min_clusters, max_clusters + 1):
    # Create KMeans instance and fit the model
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)

    # Append the WCSS to the list
    wcss.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range(min_clusters, max_clusters + 1), wcss, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.xticks(np.arange(min_clusters, max_clusters + 1, step=1))
plt.grid(True)
plt.show()
