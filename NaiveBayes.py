import numpy as np


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}

    def fit(self, X, y):
        """
        训练 Naive Bayes 模型
        :param X: 训练数据集，形状为 (n_samples, n_features)
        :param y: 类别标签，形状为 (n_samples,)
        """
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.class_priors = {cls: count / len(y) for cls, count in zip(self.classes, class_counts)}

        # 初始化特征概率字典
        self.feature_probs = {cls: {} for cls in self.classes}

        # 计算每个类别下每个特征的条件概率
        for cls in self.classes:
            cls_indices = np.where(y == cls)[0]
            X_cls = X[cls_indices]
            for i in range(X.shape[1]):
                feature_vals, feature_counts = np.unique(X_cls[:, i], return_counts=True)
                self.feature_probs[cls][i] = {val: (count / len(X_cls)) for val, count in
                                              zip(feature_vals, feature_counts)}

    def predict(self, X):
        """
        对新样本进行预测
        :param X: 新样本数据，形状为 (n_samples, n_features)
        :return: 预测类别，形状为 (n_samples,)
        """
        predictions = []
        for x in X:
            class_probs = {}
            for cls in self.classes:
                prob = self.class_priors[cls]  # 初始化为先验概率
                for i, val in enumerate(x):
                    prob *= self.feature_probs[cls][i].get(val, 1e-6)  # 使用极小值避免零概率
                class_probs[cls] = prob
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)

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


# 测试 Naive Bayes 实现
if __name__ == "__main__":
    # 训练数据集（假设离散特征）
    X_train = np.array([[1, 1], [1, 0], [0, 1], [0, 0], [1, 1]])
    y_train = np.array([1, 1, 0, 0, 1])
    # 测试数据集
    X_test = np.array([[1, 0], [0, 1]])
    y_test = np.array([1, 0])

    # 训练和评估 Naive Bayes 模型
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    metrics = nb.evaluate(y_test, y_pred)
    print(f"Model metrics:\n{metrics}")
