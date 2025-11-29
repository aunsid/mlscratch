"""
author: Aun

A simple implementation of DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

DBSCAN groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions.
It requires two parameters:
- eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- min_samples: The number of samples (or total weight) in a neighborhood for
    a point to be considered as a core point. This includes the point itself.


The algorithm works as follows:
1. For each point in the dataset, find all points within distance eps.
2. If the number of points found is greater than or equal to min_samples, mark this point as a core point.
3. For each core point, form a cluster by recursively adding all density-reachable points.
4. Points that are not reachable from any core point are labeled as noise.
"""
from collections import deque
from sklearn.datasets import make_moons, make_blobs
import numpy as np
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps=0.01, min_samples=4):
        self.eps = eps
        self.min_samples = min_samples
        self._labels = None

    def fit(self, X):
        # initialize
        num_samples, _ = X.shape

        # labels:
        # -2 - unprocessed
        # -1 - noise
        # 0, 1, ..... - class
        self._labels = np.full(num_samples, -2)
        cluster_id = -1

        for pt in range(num_samples):
            # if point processed, do nothing
            if self._labels[pt] != -2: continue

            # get neighbors
            neighbors = self.get_neighbors(X, pt)

            if len(neighbors) >= self.min_samples:
                # mark as core point
                cluster_id += 1
                self._labels[pt] = cluster_id
                # expand cluster
                self.expand_cluster(X, neighbors, cluster_id)
            else:
                # mark as noise
                self._labels[pt] = -1

    def get_neighbors(self, X, idx):
        # calculate the distance of all sample(X) to pt
        pt = X[idx]
        distances = np.linalg.norm(X - pt, axis=1)

        # return points that are closer than self.eps
        return np.where(distances<=self.eps)[0]
    
    def expand_cluster(self, X, neighbors, cluster_id):
        # bfs
        q = deque(neighbors)

        while q:
            curr = q.popleft()

            # if previously noise mark it category
            if self._labels[curr] == -1:
                self._labels[curr] = cluster_id

            # if previously seen, next pt
            if self._labels[curr] != -2:
                continue
            
            self._labels[curr] = cluster_id
            neighbors = self.get_neighbors(X, curr)

            # mark as core
            if len(neighbors) >= self.min_samples:
                for neighbor_idx in neighbors:
                    if self._labels[neighbor_idx] in [-2, -1]:
                        q.append(neighbor_idx)
                

    
    def fit_predict(self, X):
        self.fit(X)
        return self._labels

if __name__ == "__main__":
    X, _ = make_moons(n_samples=500, noise=0.05, random_state=42)

    # --- DBSCAN ---
    # Parameters tuned for density: eps=0.3 to connect the moon points, min_samples=5 
    dbscan = DBSCAN(eps=0.15, min_samples=5) 
    dbscan_labels = dbscan.fit_predict(X)
    n_clusters_dbscan = len(np.unique(dbscan_labels[dbscan_labels >= 0]))
    n_noise_dbscan = np.sum(dbscan_labels == -1)


    # DBSCAN Plot
    plt.figure(figsize=(5, 5))
    # DBSCAN plots noise (label -1) as a distinct color (grey/black)
    plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=50, alpha=0.8)
    plt.title(f'DBSCAN - Two Moons (Clusters: {n_clusters_dbscan}, Noise: {n_noise_dbscan})', fontsize=12)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()




