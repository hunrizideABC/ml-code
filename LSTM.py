import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 初始化权重
        self.W_i = np.random.randn(hidden_dim, input_dim)  # 输入到输入门的权重
        self.W_f = np.random.randn(hidden_dim, input_dim)  # 输入到遗忘门的权重
        self.W_o = np.random.randn(hidden_dim, input_dim)  # 输入到输出门的权重
        self.W_c = np.random.randn(hidden_dim, input_dim)  # 输入到候选值的权重

        self.U_i = np.random.randn(hidden_dim, hidden_dim)  # 隐藏状态到输入门的权重
        self.U_f = np.random.randn(hidden_dim, hidden_dim)  # 隐藏状态到遗忘门的权重
        self.U_o = np.random.randn(hidden_dim, hidden_dim)  # 隐藏状态到输出门的权重
        self.U_c = np.random.randn(hidden_dim, hidden_dim)  # 隐藏状态到候选值的权重

        self.b_i = np.zeros(hidden_dim)  # 输入门偏置
        self.b_f = np.zeros(hidden_dim)  # 遗忘门偏置
        self.b_o = np.zeros(hidden_dim)  # 输出门偏置
        self.b_c = np.zeros(hidden_dim)  # 候选值偏置

    def forward(self, x, h_prev, c_prev):
        """
        前向传播
        :param x: 输入 (batch_size, input_dim)
        :param h_prev: 上一时间步的隐藏状态 (batch_size, hidden_dim)
        :param c_prev: 上一时间步的单元状态 (batch_size, hidden_dim)
        :return: 当前时间步的隐藏状态和单元状态
        """
        i = sigmoid(np.dot(x, self.W_i.T) + np.dot(h_prev, self.U_i.T) + self.b_i)  # 输入门
        f = sigmoid(np.dot(x, self.W_f.T) + np.dot(h_prev, self.U_f.T) + self.b_f)  # 遗忘门
        o = sigmoid(np.dot(x, self.W_o.T) + np.dot(h_prev, self.U_o.T) + self.b_o)  # 输出门
        c_hat = np.tanh(np.dot(x, self.W_c.T) + np.dot(h_prev, self.U_c.T) + self.b_c)  # 候选值

        c = f * c_prev + i * c_hat  # 单元状态
        h = o * np.tanh(c)  # 隐藏状态

        return h, c

class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.lstm_cell = LSTMCell(input_dim, hidden_dim)
        self.W_hy = np.random.randn(output_dim, hidden_dim)  # 隐藏层到输出层的权重
        self.b_y = np.zeros(output_dim)  # 输出层偏置

    def forward(self, x):
        """
        前向传播
        :param x: 输入序列 (seq_len, batch_size, input_dim)
        :return: 输出序列 (seq_len, batch_size, output_dim)
        """
        seq_len, batch_size, _ = x.shape
        hidden_dim = self.lstm_cell.hidden_dim
        h = np.zeros((batch_size, hidden_dim))
        c = np.zeros((batch_size, hidden_dim))
        outputs = []

        for t in range(seq_len):
            x_t = x[t]
            h, c = self.lstm_cell.forward(x_t, h, c)
            y = np.dot(h, self.W_hy.T) + self.b_y
            outputs.append(y)

        return np.stack(outputs)

input_dim = 4
hidden_dim = 5
output_dim = 3
seq_len = 10
batch_size = 2

lstm = LSTM(input_dim, hidden_dim, output_dim)
X = np.random.randn(seq_len, batch_size, input_dim)  # 输入序列
output = lstm.forward(X)
print("RNN output:", output)
