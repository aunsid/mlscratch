"""
@author: Aun

Implementation of KNearestNeighbors. Based on cs231n assignments.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter 


class KNearestNeighbors:
    """Knn classifier based on L2 distance"""
    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)
    
    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sum((X[i] - self.X_train[j])**2) ** 0.5

        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i,:] = ((X[i] - self.X_train) ** 2).sum(axis=-1) ** 0.5
        
        return dists
    
    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        diff = X[:,np.newaxis,:] - self.X_train[np.newaxis,:,:]
        dists = (diff**2).sum(axis=-1)**0.5
    
        return dists
    
    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            
            # get the k nearest neighbors
            indices = np.argsort(dists[i])[:k]
            # get the labels for closests
            closest_y = [self.y_train[idx].item() for idx in indices]
            # count the votes
            counts = Counter(closest_y)
            # find the label with the maximum vote count
            most_common =counts.most_common()
            max_count = most_common[0][1]

            # tiebreak choose the numerically smallest label
            candidates = [label for label, count in most_common if count == max_count]
            candidates.sort()
            y_pred[i] = candidates[0]

        return y_pred
    
def create_toy_data():
    """Generates a simple 2D dataset for classification."""
    np.random.seed(0)
    
    # Class 0 data: Centered around (2, 2)
    X0 = np.random.randn(50, 2) * 1.5 + np.array([2, 2])
    y0 = np.zeros(50)
    
    # Class 1 data: Centered around (7, 7)
    X1 = np.random.randn(50, 2) * 1.5 + np.array([7, 7])
    y1 = np.ones(50)
    
    # Combine and shuffle
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Shuffle the data
    permutation = np.random.permutation(len(X))
    X, y = X[permutation], y[permutation]
    
    return X, y
    
def plot_decision_boundary(knn_classifier, X_train, y_train, k):
    """Plots the decision boundary created by the KNN classifier."""
    
    # Define the range for the plot
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    
    # Create a grid of points (the background test data)
    h = 0.1 # Step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict the labels for every point on the grid
    # Reshape the grid points into a test set: (num_points, 2)
    Z = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()], k=k, num_loops=0)
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

    # Plot the training points
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                          edgecolor='k', s=80, cmap=plt.cm.coolwarm)
    
    plt.title(f"k-Nearest Neighbors Decision Boundary (k={k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    # Create legend for the classes
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes", loc="lower left")
    plt.gca().add_artist(legend1)
    
    plt.show()



if __name__ == "__main__":

    # 1. Setup Data
    X_train, y_train = create_toy_data()
    
    # 2. Instantiate and Train
    knn = KNearestNeighbors()
    knn.train(X_train, y_train)
    
    print("K-Nearest Neighbors Classifier Demo")
    print(f"Total training points: {len(X_train)}")
    print(f"Features dimension: {X_train.shape[1]}")
    
    # 3. Test a single point manually
    test_point = np.array([[5, 4]])
    
    # Set k=3 for classification
    k_value = 3
    
    # Predict the label for the test point
    prediction = knn.predict(test_point, k=k_value, num_loops=0)
    
    print("-" * 30)
    print(f"Test Point: {test_point[0]}")
    print(f"Predicted Class (k={k_value}): {int(prediction)}")
    print("-" * 30)

    # 4. Generate the plot showing the decision boundary
    # This calculation predicts labels for thousands of points (the mesh grid)
    plot_decision_boundary(knn, X_train, y_train, k=k_value)


    # testing differences between distances
    np.random.seed(42)
    num_train = 200
    num_test = 50
    D = 10 # Feature Dimension

    X_train = np.random.randn(num_train, D)
    y_train = np.random.randint(0, 3, size=num_train)
    X_test = np.random.randn(num_test, D)

    # 2. Instantiate and Train
    knn = KNearestNeighbors()
    knn.train(X_train, y_train)

    # 3. Calculate Distances
    dists_two = knn.compute_distances_two_loops(X_test)
    dists_one = knn.compute_distances_one_loop(X_test)
    dists_no = knn.compute_distances_no_loops(X_test)

    # 4. Compare the results
    print("1-Loop vs 2-Loop check (np.allclose):", np.allclose(dists_one, dists_two))
    print("0-Loop vs 2-Loop check (np.allclose):", np.allclose(dists_no, dists_two))
    print("\nDistance matrices have the same shape:", dists_no.shape)






    
    
