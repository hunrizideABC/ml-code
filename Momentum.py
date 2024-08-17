import numpy as np


class Momentum:
    def __init__(self, alpha=0.01, beta=0.9, num_iterations=1000):
        self.alpha = alpha
        self.beta = beta
        self.num_iterations = num_iterations

    def fit(self, X, y, theta):
        m = len(y)
        velocity = np.zeros_like(theta)
        for _ in range(self.num_iterations):
            predictions = X.dot(theta)
            errors = predictions - y
            gradients = (1/m) * X.T.dot(errors)
            velocity = self.beta * velocity + (1 - self.beta) * gradients
            theta -= self.alpha * velocity
        return theta


# Example usage
X = np.array([[1, 1], [1, 2], [1, 3]])  # Add a column of ones for the intercept term
y = np.array([1, 2, 3])
theta = np.zeros(X.shape[1])
momentum = Momentum(alpha=0.01, beta=0.9, num_iterations=1000)
theta = momentum.fit(X, y, theta)
print("Momentum:", theta)
