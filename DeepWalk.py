import numpy as np
import networkx as nx
from gensim.models import Word2Vec


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, embedding_dim, window_size, num_epochs):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_epochs = num_epochs
        self.model = None

    def random_walk(self, start_node):
        """从指定节点开始执行随机游走"""
        walk = [start_node]
        for _ in range(self.walk_length - 1):
            current_node = walk[-1]
            neighbors = list(self.graph.neighbors(current_node))
            if len(neighbors) > 0:
                next_node = np.random.choice(neighbors)
                walk.append(next_node)
            else:
                break
        return walk

    def generate_walks(self):
        """生成所有随机游走序列"""
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(node))
        return walks

    def train(self):
        """使用 Skip-Gram 模型训练节点嵌入"""
        walks = self.generate_walks()
        walks = [[str(node) for node in walk] for walk in walks]  # 将节点转换为字符串

        # 使用Word2Vec训练
        self.model = Word2Vec(sentences=walks, vector_size=self.embedding_dim, window=self.window_size, sg=1, hs=0,
                              workers=4, epochs=self.num_epochs)

    def get_embedding(self, node):
        """获取节点的嵌入表示"""
        if self.model:
            return self.model.wv[str(node)]
        else:
            return None


if __name__ == "__main__":
    # 创建一个简单的图
    G = nx.karate_club_graph()
    # 创建DeepWalk实例
    deepwalk = DeepWalk(graph=G, walk_length=10, num_walks=80, embedding_dim=64, window_size=5, num_epochs=5)
    # 训练模型
    deepwalk.train()
    # 获取节点的嵌入表示
    node = 0
    embedding = deepwalk.get_embedding(node)
    print(f"Node {node} embedding:\n{embedding}")
