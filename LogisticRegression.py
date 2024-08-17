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

    def evaluate(self, y_true, y_pred):
        """计算并输出模型的准确率、精确率、召回率和F1分数"""
        accuracy = np.mean(y_true == y_pred)

        # True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == -1) & (y_pred == 1))
        TN = np.sum((y_true == -1) & (y_pred == -1))
        FN = np.sum((y_true == 1) & (y_pred == -1))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

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
    metrics = model.evaluate(y, predictions)
    print(f"Model metrics:\n{metrics}")
