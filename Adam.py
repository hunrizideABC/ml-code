import numpy as np


class Adam:
    def __init__(self, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=1000):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.num_iterations = num_iterations

    def fit(self, X, y, theta):
        m = len(y)
        m_t = np.zeros_like(theta)
        v_t = np.zeros_like(theta)
        for t in range(1, self.num_iterations + 1):
            predictions = X.dot(theta)
            errors = predictions - y
            gradients = (1 / m) * X.T.dot(errors)

            m_t = self.beta1 * m_t + (1 - self.beta1) * gradients
            v_t = self.beta2 * v_t + (1 - self.beta2) * gradients ** 2

            m_t_hat = m_t / (1 - self.beta1 ** t)
            v_t_hat = v_t / (1 - self.beta2 ** t)

            theta -= self.alpha * m_t_hat / (np.sqrt(v_t_hat) + self.epsilon)
        return theta


# Example usage
X = np.array([[1, 1], [1, 2], [1, 3]])  # Add a column of ones for the intercept term
y = np.array([1, 2, 3])
theta = np.zeros(X.shape[1])
adam = Adam(alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=1000)
theta = adam.fit(X, y, theta)
print("Adam:", theta)
