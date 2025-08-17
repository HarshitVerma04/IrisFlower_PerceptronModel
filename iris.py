import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Select only two classes: Setosa (0) and Versicolor (1)
binary_mask = y < 2
X = X[binary_mask]
y = y[binary_mask]

# Use only two features: sepal length (0) and petal length (2)
X = X[:, [0, 2]]

# Add bias term (1) to X
X = np.c_[np.ones(X.shape[0]), X]

def perceptron_train(X, y, lr=0.01, epochs=20):
    w = np.zeros(X.shape[1])  # Initialize weights to zero
    for epoch in range(epochs):
        for xi, target in zip(X, y):
            update = lr * (target - predict(xi, w))
            w += update * xi
    return w
def predict(x, w):
    return 1 if np.dot(x, w) >= 0 else 0

def plot_decision_boundary(X, y, weights):
    x_min, x_max = X[:,1].min() - 1, X[:,1].max() + 1
    y_min, y_max = X[:,2].min() - 1, X[:,2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
    zz = np.array([predict(xi, weights) for xi in grid])
    zz = zz.reshape(xx.shape)

    plt.contourf(xx, yy, zz, alpha=0.4)
    plt.scatter(X[:,1], X[:,2], c=y, edgecolor='k', marker='o')
    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Length")
    plt.title("Perceptron Decision Boundary")
    plt.show()

plot_decision_boundary(X, y, weights)

