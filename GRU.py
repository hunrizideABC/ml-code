import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class GRUCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 初始化权重
        self.W_z = np.random.randn(hidden_dim, input_dim)  # 重置门的输入权重
        self.W_r = np.random.randn(hidden_dim, input_dim)  # 更新门的输入权重
        self.W_h = np.random.randn(hidden_dim, input_dim)  # 计算候选隐藏状态的输入权重

        self.U_z = np.random.randn(hidden_dim, hidden_dim)  # 重置门的隐藏状态权重
        self.U_r = np.random.randn(hidden_dim, hidden_dim)  # 更新门的隐藏状态权重
        self.U_h = np.random.randn(hidden_dim, hidden_dim)  # 计算候选隐藏状态的隐藏状态权重

        self.b_z = np.zeros(hidden_dim)  # 重置门偏置
        self.b_r = np.zeros(hidden_dim)  # 更新门偏置
        self.b_h = np.zeros(hidden_dim)  # 计算候选隐藏状态偏置

    def forward(self, x, h_prev):
        """
        前向传播
        :param x: 输入 (batch_size, input_dim)
        :param h_prev: 上一时间步的隐藏状态 (batch_size, hidden_dim)
        :return: 当前时间步的隐藏状态
        """
        z = sigmoid(np.dot(x, self.W_z.T) + np.dot(h_prev, self.U_z.T) + self.b_z)  # 重置门
        r = sigmoid(np.dot(x, self.W_r.T) + np.dot(h_prev, self.U_r.T) + self.b_r)  # 更新门
        h_hat = np.tanh(np.dot(x, self.W_h.T) + np.dot(r * h_prev, self.U_h.T) + self.b_h)  # 候选隐藏状态
        h = (1 - z) * h_prev + z * h_hat  # 当前隐藏状态

        return h


class GRU:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.gru_cell = GRUCell(input_dim, hidden_dim)
        self.W_hy = np.random.randn(output_dim, hidden_dim)  # 隐藏层到输出层的权重
        self.b_y = np.zeros(output_dim)  # 输出层偏置

    def forward(self, x):
        """
        前向传播
        :param x: 输入序列 (seq_len, batch_size, input_dim)
        :return: 输出序列 (seq_len, batch_size, output_dim)
        """
        seq_len, batch_size, _ = x.shape
        hidden_dim = self.gru_cell.hidden_dim
        output_dim = self.W_hy.shape[0]

        h = np.zeros((batch_size, hidden_dim))
        outputs = []

        for t in range(seq_len):
            x_t = x[t]
            h = self.gru_cell.forward(x_t, h)
            y = np.dot(h, self.W_hy.T) + self.b_y
            outputs.append(y)

        return np.stack(outputs)


input_dim = 4
hidden_dim = 5
output_dim = 3
seq_len = 10
batch_size = 2

gru = GRU(input_dim, hidden_dim, output_dim)
X = np.random.randn(seq_len, batch_size, input_dim)  # 输入序列
output = gru.forward(X)
print("RNN output:", output)
