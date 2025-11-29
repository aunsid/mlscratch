"""
author: Aun

A simple implementation of Logistic Regression using manual backpropagation.
y = sigmoid(XW + b)

The loss function used is Binary Cross Entropy (BCE):
L = - (1/N) * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
"""

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def  __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, reg_lambda=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        assert regularization in (None, 'l1', 'l2'), "Regularization must be None, 'l1', or 'l2'"
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # initalize
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features, 1) * 0.1
        self.bias = 0

        for it in range(self.n_iterations):
            # Forward pass
            h = X @ self.weights + self.bias
            y_predicted = self.sigmoid(h)

            # calculate the loss
            loss = (-1.0 / n_samples) * np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1-y_predicted))
            self.losses.append(loss)

            if it % 100 == 0:
                print(f"Iteration {it}, Loss: {loss}")

            # Add regularization to loss
            if self.regularization == 'l2':
                loss += (self.reg_lambda / (2 * n_samples)) * np.sum(self.weights ** 2)
            elif self.regularization == 'l1':
                loss += (self.reg_lambda / (2 * n_samples)) * np.sum(np.abs(self.weights))  
            else:
                pass

            # Compute gradients
            dloss = 1
            dy_predicted = -((y / y_predicted) - (1 - y) / (1 - y_predicted)) * dloss
            dh =y_predicted * (1 - y_predicted) * dy_predicted
            dweights = (1/n_samples) * (X.T @ dh)
            dbias = (1/n_samples) * np.sum(dh)
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
        linear_model = X @ self.weights + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    def plot_loss(self):
        plt.plot(range(self.n_iterations), self.losses)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss over Iterations")
        plt.show()

if __name__ == "__main__":
    # Example usage
    # create a simple dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])
    y = y.reshape(-1, 1)
    

    # initialize logistic regression model
    lr = LogisticRegression(learning_rate=0.01, n_iterations=1000, regularization='l2', reg_lambda=0.1)

    # train model on dataset
    lr.fit(X, y)

    # plot decision boundary    
    x1 = np.linspace(0, 6, 100)
    x2 = np.linspace(0, 8, 100)
    xx, yy = np.meshgrid(x1, x2)
    Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # plot data points
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)

    plt.show()

    # plot loss over iterations
    lr.plot_loss()




