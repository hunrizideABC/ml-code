import numpy as np


class TextCNN:
    def __init__(self, input_dim, num_filters, filter_sizes, num_classes):
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes

        # 初始化卷积核权重和偏置
        self.W = [np.random.randn(num_filters, filter_size, input_dim) for filter_size in filter_sizes]
        self.b = [np.zeros(num_filters) for _ in filter_sizes]

        # 初始化全连接层权重和偏置
        self.W_fc = np.random.randn(num_filters * len(filter_sizes), num_classes)
        self.b_fc = np.zeros(num_classes)

    def conv1d(self, x, W, b):
        """
        一维卷积操作
        :param x: 输入 (batch_size, seq_len, input_dim)
        :param W: 卷积核 (num_filters, filter_size, input_dim)
        :param b: 偏置 (num_filters)
        :return: 卷积结果 (batch_size, conv_len, num_filters)
        """
        batch_size, seq_len, input_dim = x.shape
        num_filters, filter_size, _ = W.shape
        conv_len = seq_len - filter_size + 1
        conv_out = np.zeros((batch_size, conv_len, num_filters))
        for i in range(conv_len):
            x_slice = x[:, i:i + filter_size, :]
            for j in range(num_filters):
                conv_out[:, i, j] = np.sum(x_slice * W[j, :, :], axis=(1, 2)) + b[j]
        return conv_out

    def max_pool1d(self, x, pool_size):
        """
        一维最大池化操作
        :param x: 输入 (batch_size, seq_len, num_filters)
        :param pool_size: 池化窗口大小
        :return: 池化结果 (batch_size, new_seq_len, num_filters)
        """
        batch_size, seq_len, num_filters = x.shape
        new_seq_len = (seq_len - pool_size) + 1
        pool_out = np.zeros((batch_size, new_seq_len, num_filters))
        for i in range(new_seq_len):
            pool_out[:, i, :] = np.max(x[:, i:i + pool_size, :], axis=1)
        return pool_out

    def forward(self, x):
        """
        前向传播
        :param x: 输入 (batch_size, seq_len, input_dim)
        :return: 预测结果 (batch_size, num_classes)
        """
        conv_outs = []
        for i in range(len(self.filter_sizes)):
            conv_out = self.conv1d(x, self.W[i], self.b[i])
            pool_out = self.max_pool1d(conv_out, conv_out.shape[1])
            conv_outs.append(pool_out)
        conv_outs = np.concatenate(conv_outs, axis=2)
        flattened = conv_outs.reshape(conv_outs.shape[0], -1)
        logits = np.dot(flattened, self.W_fc) + self.b_fc
        return logits


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


seq_len = 15
input_dim = 10
num_filters = 5
filter_sizes = [3, 4, 5]
num_classes = 3
X = np.random.randn(10, seq_len, input_dim)
textcnn = TextCNN(input_dim, num_filters, filter_sizes, num_classes)
logits = textcnn.forward(X)
print(logits)
