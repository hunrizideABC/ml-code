import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        DBSCAN 聚类算法
        :param eps: 邻域半径
        :param min_samples: 每个核心点的最小邻域点数
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        """
        训练 DBSCAN 模型
        :param X: 训练数据集，形状为 (n_samples, n_features)
        """
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)  # -1 表示噪声

        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # 标记为噪声
            else:
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1

    def _region_query(self, X, point_idx):
        """
        查找指定点的邻域内的点
        :param X: 数据集
        :param point_idx: 点的索引
        :return: 邻域点的索引
        """
        distances = pairwise_distances(X[point_idx].reshape(1, -1), X, metric='euclidean').flatten()
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        """
        扩展簇
        :param X: 数据集
        :param point_idx: 核心点的索引
        :param neighbors: 核心点的邻域
        :param cluster_id: 当前簇的 ID
        """
        self.labels_[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = self._region_query(X, neighbor_idx)
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, neighbor_neighbors])
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
            i += 1

    def fit_predict(self, X):
        """
        训练模型并对数据进行预测
        :param X: 训练数据集
        :return: 聚类标签，形状为 (n_samples,)
        """
        self.fit(X)
        return self.labels_

# 测试 DBSCAN 实现
if __name__ == "__main__":
    # 创建简单数据集
    X = np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
        [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0],
    ])

    # 训练和预测 DBSCAN 模型
    dbscan = DBSCAN(eps=3, min_samples=2)
    labels = dbscan.fit_predict(X)

    print("Labels:", labels)
