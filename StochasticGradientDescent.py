import numpy as np


class StochasticGradientDescent:
    def __init__(self, alpha=0.01, num_epochs=100):
        self.alpha = alpha
        self.num_epochs = num_epochs

    def fit(self, X, y, theta):
        m = len(y)
        for _ in range(self.num_epochs):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                predictions = xi.dot(theta)
                error = predictions - yi
                gradients = xi.T.dot(error)
                theta -= self.alpha * gradients
        return theta


# Example usage
X = np.array([[1, 1], [1, 2], [1, 3]])  # Add a column of ones for the intercept term
y = np.array([1, 2, 3])
theta = np.zeros(X.shape[1])
sgd = StochasticGradientDescent(alpha=0.01, num_epochs=100)
theta = sgd.fit(X, y, theta)
print("SGD:", theta)
