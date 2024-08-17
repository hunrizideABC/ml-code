import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    def softmax_derivative(self, x):
        s = self.softmax(x)
        return s * (1 - s)

    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def forward(self, X):
        # Forward pass
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = (self.A1.T.dot(dZ2)) / m
        db2 = np.sum(dZ2, axis=0) / m

        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (X.T.dot(dZ1)) / m
        db1 = np.sum(dZ1, axis=0) / m

        # Update weights and biases
        self.W1 -= 0.01 * dW1
        self.b1 -= 0.01 * db1
        self.W2 -= 0.01 * dW2
        self.b2 -= 0.01 * db2


# Example usage
input_size = 3
hidden_size = 5
output_size = 2

nn = NeuralNetwork(input_size, hidden_size, output_size)
X = np.array([[1, 2, 3]])
y = np.array([[0, 1]])  # One-hot encoded target
output = nn.forward(X)
print("Forward pass output:", output)

nn.backward(X, y)
print("Updated weights and biases:")
print("W1:", nn.W1)
print("b1:", nn.b1)
print("W2:", nn.W2)
print("b2:", nn.b2)
