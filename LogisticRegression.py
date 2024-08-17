import numpy as np

class LogisticRegression:
    def __init__(self, alpha=0.01, num_iters=1000):
        self.alpha = alpha
        self.num_iters = num_iters
        self.theta = None

    def sigmoid(self, z):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        """Compute the cost function."""
        m = len(y)
        h = self.sigmoid(np.dot(X, self.theta))
        cost = (-1/m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
        return cost

    def gradient_descent(self, X, y):
        """Perform gradient descent to learn theta."""
        m = len(y)
        self.theta = np.zeros(X.shape[1])
        cost_history = []

        for _ in range(self.num_iters):
            h = self.sigmoid(np.dot(X, self.theta))
            gradient = np.dot(X.T, (h - y)) / m
            self.theta -= self.alpha * gradient
            cost = self.compute_cost(X, y)
            cost_history.append(cost)

        return self.theta, cost_history

    def fit(self, X, y):
        """Fit the model to the data."""
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        self.theta, self.cost_history = self.gradient_descent(X, y)
        return self

    def predict(self, X):
        """Make predictions using the learned model."""
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        probabilities = self.sigmoid(np.dot(X, self.theta))
        return (probabilities >= 0.5).astype(int)

    def accuracy(self, y_true, y_pred):
        """Calculate the accuracy of the model."""
        return np.mean(y_true == y_pred) * 100

# 使用示例
if __name__ == "__main__":
    # 1. 数据准备
    np.random.seed(42)
    X = np.random.rand(100, 2)  # 100个样本，每个样本有两个特征
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # 简单的标签规则，x1 + x2 > 1的样本属于类别1，否则属于类别0

    # 2. 创建逻辑回归模型实例
    model = LogisticRegression(alpha=0.01, num_iters=1000)

    # 3. 训练模型
    model.fit(X, y)

    # 4. 预测
    predictions = model.predict(X)

    # 5. 评估
    accuracy = model.accuracy(y, predictions)
    print(f"Trained theta: {model.theta}")
    print(f"Model accuracy: {accuracy:.2f}%")
