"""
@author: Aun

A simple implementation of Linear Regression using manual backpropagation.
y = XW + b

The loss function used is Mean Squared Error (MSE):
L = (1/2N) * Σ(y_pred - y_true)^2
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, reg_lambda=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        assert regularization in (None, 'l1', 'l2'), "Regularization must be None, 'l1', or 'l2'"
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        # initialize
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features, 1)
        self.bias = 0 

        self.losses = []
        for it in range(self.n_iterations):
            # Forward pass
            y_predicted = X @ self.weights + self.bias
            error = y_predicted - y 
            loss = (1 / (2 * n_samples)) * np.sum(error ** 2)
            
            # Add regularization to loss
            if self.regularization == 'l2':
                loss += (self.reg_lambda / (2 * n_samples)) * np.sum(self.weights ** 2)
            elif self.regularization == 'l1':
                loss += (self.reg_lambda / (2 * n_samples)) * np.sum(np.abs(self.weights))  
            else:
                pass

            self.losses.append(loss)

            if it % 100 == 0:
                print(f"Iteration {it}, Loss: {loss}")

            # Compute gradients
            dloss = 1.0
            derror = (1/n_samples) * error * dloss
            dweights = X.T @ derror
            dbias = np.sum(derror)

            # Add regularization to gradients
            if self.regularization == 'l2':
                dweights += (self.reg_lambda / n_samples) * self.weights
            elif self.regularization == 'l1':
                dweights += (self.reg_lambda / n_samples) * np.sign(self.weights)
            else:
                pass

            # Update weights and bias
            self.weights -= self.learning_rate * dweights
            self.bias -= self.learning_rate * dbias
    
    def predict(self, X):
        return X @ self.weights + self.bias


if __name__ == "__main__":
    # Example usage

    # create a simple dataset
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 3 * X + 7 + np.random.randn(100, 1) * 2  # y = 3x + 7 + noise

    # Train the model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000, regularization='l2', reg_lambda=0.1)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', label='Predicted function')
    plt.plot(X, 3 * X + 7, color='green', label='True function: y=3x+7')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(model.losses)), model.losses, color='purple')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss over Iterations')
    plt.show() 