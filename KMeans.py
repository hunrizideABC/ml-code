import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        """
        K-Means 聚类算法
        :param n_clusters: 簇的数量
        :param max_iter: 最大迭代次数
        :param tol: 当质心的变化小于该阈值时，停止迭代
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        """
        训练 K-Means 模型
        :param X: 训练数据集，形状为 (n_samples, n_features)
        """
        # 随机初始化质心
        np.random.seed(42)  # 为了结果可重复
        initial_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[initial_indices]

        for i in range(self.max_iter):
            # 分配每个数据点到最近的质心
            labels = self._assign_labels(X)
            # 计算新的质心
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])
            # 检查质心变化是否小于容忍度
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

    def _assign_labels(self, X):
        """
        分配每个数据点到最近的质心
        :param X: 数据集
        :return: 每个数据点的标签，形状为 (n_samples,)
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        """
        对新数据进行聚类
        :param X: 新数据集
        :return: 聚类标签，形状为 (n_samples,)
        """
        return self._assign_labels(X)

    def fit_predict(self, X):
        """
        训练模型并对训练数据进行预测
        :param X: 训练数据集
        :return: 聚类标签，形状为 (n_samples,)
        """
        self.fit(X)
        return self.predict(X)


# 测试 K-Means 实现
if __name__ == "__main__":
    # 创建简单数据集
    X = np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
        [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0],
    ])

    # 训练和预测 K-Means 模型
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(X)

    print("Cluster Centers:\n", kmeans.centroids)
    print("Labels:", labels)
