import numpy as np


def max_pool1d(x, pool_size):
    """
    一维最大池化操作
    :param x: 输入 (batch_size, seq_len, num_filters)
    :param pool_size: 池化窗口大小
    :return: 池化结果 (batch_size, (seq_len - pool_size + 1), num_filters)
    """
    batch_size, seq_len, num_filters = x.shape
    pool_out = np.zeros((batch_size, seq_len - pool_size + 1, num_filters))

    for i in range(seq_len - pool_size + 1):
        pool_out[:, i, :] = np.max(x[:, i:i + pool_size, :], axis=1)

    return pool_out


# 示例数据
x = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]],
              [[9, 10], [11, 12], [13, 14], [15, 16]]])  # (2, 4, 2) batch_size=2, seq_len=4, num_filters=2
pool_size = 2

# 进行最大池化
pooled_output = max_pool1d(x, pool_size)
print("Pooled Output:")
print(pooled_output)
