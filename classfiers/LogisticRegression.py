import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=10):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.label_type = None  # 初始化标签类型
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        X: 数据矩阵，形状为 (N, d)
        y: 标签向量，形状为 (N,) 且取值为 {0, 1} 或 {-1, +1}
        """
        # 判断标签类型并设置 label_type
        if np.array_equal(np.unique(y), [0, 1]):
            self.label_type = 'binary_01'  # 标签为 {0, 1}
            y = 2 * y - 1  # 将 {0, 1} 转换为 {-1, +1}
        elif np.array_equal(np.unique(y), [-1, 1]):
            self.label_type = 'binary_pm1'  # 标签为 {-1, +1}
        else:
            raise ValueError("Labels must be either {0, 1} or {-1, +1}.")

        N, d = X.shape
        self.w = np.zeros(d)  # 初始化权重为零

        for i in range(self.num_iterations):
            z = X.dot(self.w)  # 形状为 (N,)
            yz = y * z  # 形状为 (N,)
            # 计算 sigmoid(-y * (Xw))
            sigma = self.sigmoid(-yz)  # 形状为 (N,)
            # 计算梯度
            gradient = -X.T.dot(y * sigma)  # 形状为 (d,)
            # 更新权重
            self.w -= self.learning_rate * gradient

            # 计算并打印损失
            loss = np.sum(np.log(1 + np.exp(-yz)))
            self.loss_history.append(loss)
            print(f'Iteration {i + 1}/{self.num_iterations}, Loss: {loss:.4f}')

    def predict_proba(self, X):
        """
        预测样本属于+1的概率
        """
        z = X.dot(self.w)
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        预测标签
        如果训练时标签是 {0, 1}，则返回 {0, 1}
        如果训练时标签是 {-1, +1}，则返回 {-1, +1}
        """
        proba = self.predict_proba(X)
        raw_predictions = np.where(proba >= threshold, 1, -1)

        # 如果标签是 {0, 1}，将预测结果从 {-1, +1} 转回 {0, 1}
        if self.label_type == 'binary_01':
            return (raw_predictions + 1) // 2  # 将 -1 转为 0，+1 转为 1
        return raw_predictions



