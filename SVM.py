import numpy as np
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iters = num_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """训练SVM模型"""
        n_samples, n_features = X.shape

        # 初始化权重和偏置
        self.w = np.zeros(n_features)
        self.b = 0

        # 将标签转换为-1和1
        y_ = np.where(y <= 0, -1, 1)

        # 梯度下降
        for _ in range(self.num_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # 正确分类，进行正则化处理
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # 错误分类，更新权重和偏置
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        """使用训练好的SVM模型进行预测"""
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

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
    # 数据准备
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100个样本，每个样本有两个特征
    y = np.array([1 if x[0] * x[1] > 0 else 0 for x in X])  # 生成标签，简单规则
    y = np.where(y == 0, -1, 1)  # 将标签转换为-1和1
    # 创建SVM实例
    model = SVM(learning_rate=0.001, lambda_param=0.01, num_iters=1000)
    # 训练模型
    model.fit(X, y)
    # 预测
    predictions = model.predict(X)
    # 评估
    metrics = model.evaluate(y, predictions)
    print(f"Model metrics:\n{metrics}")
