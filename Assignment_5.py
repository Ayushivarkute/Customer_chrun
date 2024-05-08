import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
from matplotlib import cm

dataset = pd.read_csv("C:/Users/ayush/Downloads/data.csv")
X = dataset[['Height','Weight']]

min_clusters = 2
max_clusters = 8

best_avg = -1
best_k=-1
for k in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"For n_clusters = {k}, the average silhouette_score is: {silhouette_avg}")
    if best_avg<silhouette_avg:
        best_avg = silhouette_avg
        best_k = k

print(best_avg)
kmeans = KMeans(n_clusters=best_k, random_state=0)
cluster_labels = kmeans.fit_predict(X)
cluster = kmeans.cluster_centers_
title = 'ayushi'

plt.scatter(X['Height'], X['Weight'], c=cluster_labels, s=30, cmap='viridis')
plt.scatter(cluster[:, 0], cluster[:, 1], c='red', s=200, marker='X')
plt.title(title)
plt.legend()
plt.show()