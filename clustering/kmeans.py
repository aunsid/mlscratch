"""
author: Aun

A simple implementation of K-Means Clustering.

K-Means aims to partition n samples into k clusters in which each sample belongs to the cluster with the nearest mean,
serving as a prototype of the cluster.

The algorithm works as follows:
1. Initialize k centroids randomly from the dataset.
2. Assign each data point to the nearest centroid, forming k clusters.
3. Update the centroids by calculating the mean of all data points assigned to each cluster.
4. Repeat steps 2 and 3 until convergence (i.e., centroids do not change).

The loss function used is the Sum of Squared Distances:
L = Σ ||x_i - μ_j||^2
where x_i is a data point and μ_j is the centroid of cluster j.
"""
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, n_iterations=100):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self._centroids = None
        self._labels = None

    def assign_points(self, X):
        # assign each point to the closest centroid
        distances = np.linalg.norm(X - self._centroids[:, np.newaxis], axis=-1)
        return np.argmin(distances, axis=0)
        

    def update_centroids(self, X):
        # recompute of mean of the clusters
        # these new means are the centroids

        centroids = np.zeros_like(self._centroids)

        for k in range(self.n_clusters):
            cluster_points = X[self._labels==k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                centroids[k] = X[np.random.choice(X.shape[0])]

        return centroids 

    
    def fit(self, X):
        # init
        num_samples, _ = X.shape
        # choose k cluster centers randomly from X
        random_indices = np.random.choice(num_samples, self.n_clusters, replace=False)
        # set these to be centroids
        self._centroids = X[random_indices]

        # iterate 
        for _ in range(self.n_iterations):
            old_centroids = self._centroids.copy()
            # assign labels to all points
            self._labels = self.assign_points(X)
            # get new centers
            self._centroids = self.update_centroids(X)

            # centroids did not change
            if np.all(old_centroids == self._centroids):
                break


    def fit_predict(self, X):
        self.fit(X)
        return self._labels
    

if __name__ == "__main__":
    np.random.seed(42)

    X = np.vstack([
        np.random.randn(100, 2) + [10, 10],
        np.random.randn(100, 2) + [-10, -10],
        np.random.randn(100, 2) + [-10, 10],
        np.random.randn(100, 2) + [10, -10]
    ])

    kmeans = KMeans(4, 100)
    labels = kmeans.fit_predict(X)
    centroids = kmeans._centroids


    plt.figure(figsize=(10, 10))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
    plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='X')
    plt.show()