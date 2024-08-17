import numpy as np
from scipy.spatial import distance


class KNN:
    def __init__(self, k=3):
        """
        K-Nearest Neighbors 分类器
        :param k: 最近邻居的数量
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        训练 KNN 模型
        :param X: 训练数据集，形状为 (n_samples, n_features)
        :param y: 训练数据标签，形状为 (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        对数据进行预测
        :param X: 测试数据集，形状为 (n_samples, n_features)
        :return: 预测标签，形状为 (n_samples,)
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """
        对单个样本进行预测
        :param x: 测试样本
        :return: 预测标签
        """
        # 计算 x 与训练集中所有样本的距离
        distances = distance.cdist([x], self.X_train, 'euclidean').flatten()
        # 获取最近的 k 个邻居的索引
        k_indices = np.argsort(distances)[:self.k]
        # 获取最近 k 个邻居的标签
        k_nearest_labels = self.y_train[k_indices]
        # 返回出现频率最高的标签
        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique[np.argmax(counts)]

# 测试 KNN 实现
if __name__ == "__main__":
    # 创建简单数据集
    X_train = np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
        [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0],
    ])
    y_train = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1])

    X_test = np.array([
        [1.0, 1.0],
        [7.0, 7.0],
    ])

    # 训练和预测 KNN 模型
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    print("Predictions:", predictions)
