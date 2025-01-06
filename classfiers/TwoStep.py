import numpy as np


class TwoStep():
    def __init__(self, base_classifier, base_regressor, max_iter, k):
        """
        初始化 TwoStep 类。

        参数:
        - base_classifier: 基础分类器，必须实现 `predict` 和 `predict_proba` 方法。
        - base_regressor: 基础回归器，必须实现 `predict` 方法。
        - max_iter: 最大迭代次数。
        - k: 每轮迭代选取前百分之多少的样本，范围为 (0, 1]。
        """
        self.clf = base_classifier
        self.reg = base_regressor
        self.max_iter = max_iter
        self.k = k

    def fit(self, X, y):
        """
        训练 TwoStep 模型。

        参数:
        - X: 特征矩阵 (numpy ndarray)。
        - y: 标签向量 (numpy ndarray)，其中无标签的数据用 np.nan 表示。

        返回:
        - self: 训练后的模型。
        """
        # 0. 校验 X 和 y 的类型
        if not isinstance(X, np.ndarray):
            raise TypeError("X 必须是 numpy ndarray 类型。")
        if not isinstance(y, np.ndarray):
            raise TypeError("y 必须是 numpy ndarray 类型。")

        # 校验 X 和 y 的形状是否匹配
        if X.shape[0] != y.shape[0]:
            raise ValueError("X 和 y 的样本数量 (行数) 必须一致。")

        # 1. 复制 X 和 y，避免操作原始数据
        X_copy = X.copy()
        y_copy = y.copy()

        # 2. 根据 y 划分 L 和 U
        # L: 有标签数据 (y != np.nan)
        # U: 无标签数据 (y == np.nan)
        labeled_mask = ~np.isnan(y_copy)  # 有标签数据的布尔掩码
        unlabeled_mask = np.isnan(y_copy)  # 无标签数据的布尔掩码

        X_L = X_copy[labeled_mask]  # 有标签数据的特征
        y_L = y_copy[labeled_mask]  # 有标签数据的标签

        X_U = X_copy[unlabeled_mask]  # 无标签数据的特征

        # 打印划分结果（可选）
        print(f"有标签数据 (L): {X_L.shape[0]} 样本")
        print(f"无标签数据 (U): {X_U.shape[0]} 样本")

        # 后续逻辑可以在这里继续实现，例如训练分类器和回归器
        # ...

        return self
