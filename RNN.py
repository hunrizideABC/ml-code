import numpy as np


class RNNCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_xh = np.random.randn(hidden_dim, input_dim)  # 输入到隐藏层的权重
        self.W_hh = np.random.randn(hidden_dim, hidden_dim)  # 隐藏层到隐藏层的权重
        self.b_h = np.zeros(hidden_dim)  # 隐藏层偏置

    def forward(self, x, h_prev):
        h = np.tanh(np.dot(x, self.W_xh.T) + np.dot(h_prev, self.W_hh.T) + self.b_h)
        return h


class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.rnn_cell = RNNCell(input_dim, hidden_dim)
        self.W_hy = np.random.randn(output_dim, hidden_dim)  # 隐藏层到输出层的权重
        self.b_y = np.zeros(output_dim)  # 输出层偏置

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        hidden_dim = self.rnn_cell.hidden_dim
        h = np.zeros((batch_size, hidden_dim))
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            h = self.rnn_cell.forward(x_t, h)
            y = np.dot(h, self.W_hy.T) + self.b_y
            outputs.append(y)
        return np.stack(outputs)


input_dim = 4
hidden_dim = 5
output_dim = 3
seq_len = 10
batch_size = 2

rnn = RNN(input_dim, hidden_dim, output_dim)
X = np.random.randn(seq_len, batch_size, input_dim)  # 输入序列
output = rnn.forward(X)
print("RNN output:", output)
