import numpy as np


def max_pool2d(x, pool_size):
    """
    二维最大池化操作
    :param x: 输入 (batch_size, height, width, channels)
    :param pool_size: 池化窗口大小 (pool_height, pool_width)
    :return: 池化结果 (batch_size, new_height, new_width, channels)
    """
    batch_size, height, width, channels = x.shape
    pool_height, pool_width = pool_size
    new_height = height - pool_height + 1
    new_width = width - pool_width + 1
    pool_out = np.zeros((batch_size, new_height, new_width, channels))

    for i in range(new_height):
        for j in range(new_width):
            pool_out[:, i, j, :] = np.max(x[:, i:i + pool_height, j:j + pool_width, :], axis=(1, 2))

    return pool_out


x = np.array([[[[1, 2], [3, 4]],
               [[5, 6], [7, 8]]],
              [[[9, 10], [11, 12]],
               [[13, 14], [15, 16]]]])
pool_size = (2, 2)

# 进行最大池化
pooled_output = max_pool2d(x, pool_size)
print("Pooled Output:")
print(pooled_output)
