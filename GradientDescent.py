import numpy as np


class GradientDescent:
    def __init__(self, alpha=0.01, num_iterations=1000):
        self.alpha = alpha
        self.num_iterations = num_iterations

    def fit(self, X, y, theta):
        m = len(y)
        for _ in range(self.num_iterations):
            predictions = X.dot(theta)
            errors = predictions - y
            gradients = (1/m) * X.T.dot(errors)
            theta -= self.alpha * gradients
        return theta

# Example usage
X = np.array([[1, 1], [1, 2], [1, 3]])  # Add a column of ones for the intercept term
y = np.array([1, 2, 3])
theta = np.zeros(X.shape[1])
gd = GradientDescent(alpha=0.01, num_iterations=1000)
theta = gd.fit(X, y, theta)
print("Gradient Descent:", theta)
