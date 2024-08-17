import numpy as np


class DNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # 初始化权重和偏置
        np.random.seed(42)
        self.weights_input_hidden = np.random.rand(input_dim, hidden_dim)
        self.weights_hidden_output = np.random.rand(hidden_dim, output_dim)
        self.bias_hidden = np.random.rand(hidden_dim)
        self.bias_output = np.random.rand(output_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_layer_input)
        return self.predicted_output

    def backward(self, X, y):
        output_layer_error = y - self.predicted_output
        output_layer_delta = output_layer_error * self.sigmoid_derivative(self.predicted_output)

        hidden_layer_error = np.dot(output_layer_delta, self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)

        # 更新权重和偏置
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_layer_delta) * self.learning_rate
        self.bias_output += np.sum(output_layer_delta, axis=0) * self.learning_rate

        self.weights_input_hidden += np.dot(X.T, hidden_layer_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0) * self.learning_rate

    def train(self, X_train, y_train, epochs=10000):
        for epoch in range(epochs):
            # 前向传播
            self.forward(X_train)

            # 计算损失（均方误差）
            loss = np.mean((y_train - self.predicted_output) ** 2)

            # 反向传播
            self.backward(X_train, y_train)

            # 打印损失
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        return self.forward(X)


# 示例数据
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# 创建并训练模型
input_dim = 2
hidden_dim = 4
output_dim = 1
learning_rate = 0.1

model = DNN(input_dim, hidden_dim, output_dim, learning_rate)
model.train(X_train, y_train, epochs=10000)

# 测试模型
predictions = model.predict(X_train)
print("Final predicted output:")
print(predictions)
