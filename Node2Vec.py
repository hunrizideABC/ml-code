import numpy as np
import networkx as nx
from gensim.models import Word2Vec


class Node2Vec:
    def __init__(self, graph, walk_length, num_walks, p=1, q=1, embedding_dim=64, window_size=5, num_epochs=10):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p  # Return parameter
        self.q = q  # In-out parameter
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_epochs = num_epochs
        self.model = None

    def _node2vec_walk(self, start_node):
        """执行基于 Node2Vec 策略的随机游走"""
        walk = [start_node]
        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = list(self.graph.neighbors(current))
            if len(neighbors) > 0:
                if len(walk) == 1:
                    walk.append(np.random.choice(neighbors))
                else:
                    prev = walk[-2]
                    probs = self._get_transition_probs(prev, current, neighbors)
                    next_node = np.random.choice(neighbors, p=probs)
                    walk.append(next_node)
            else:
                break
        return walk

    def _get_transition_probs(self, prev, current, neighbors):
        """计算从当前节点到下一节点的转移概率"""
        probs = []
        for neighbor in neighbors:
            if neighbor == prev:  # Return to the previous node
                probs.append(1 / self.p)
            elif self.graph.has_edge(neighbor, prev):  # Connected to previous node
                probs.append(1)
            else:  # Not connected to previous node
                probs.append(1 / self.q)
        probs = np.array(probs)
        return probs / probs.sum()

    def generate_walks(self):
        """生成所有的随机游走序列"""
        walks = []
        nodes = list(self.graph.nodes())

        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self._node2vec_walk(node))

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
    # 创建Node2Vec实例
    node2vec = Node2Vec(graph=G, walk_length=10, num_walks=80, p=1, q=2, embedding_dim=64, window_size=5, num_epochs=5)
    # 训练模型
    node2vec.train()
    # 获取节点的嵌入表示
    node = 0
    embedding = node2vec.get_embedding(node)
    print(f"Node {node} embedding:\n{embedding}")
