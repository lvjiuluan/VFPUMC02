import numpy as np
from utils.DataProcessUtils import determine_task_type
from utils.DataProcessUtils import get_top_k_percent_idx
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.info("TwoStep 类已初始化, max_iter=%d, k=%.2f", max_iter, k)

    def fit(self, X, y):
        """
        训练 TwoStep 模型。

        参数:
        - X: 特征矩阵 (numpy ndarray)。
        - y: 标签向量 (numpy ndarray)，其中无标签的数据用 np.nan 表示。

        返回:
        - self: 训练后的模型。
        """
        logging.info("开始训练 TwoStep 模型...")

        # 校验 X 和 y 的类型
        if not isinstance(X, np.ndarray):
            logging.error("X 必须是 numpy ndarray 类型。")
            raise TypeError("X 必须是 numpy ndarray 类型。")
        if not isinstance(y, np.ndarray):
            logging.error("y 必须是 numpy ndarray 类型。")
            raise TypeError("y 必须是 numpy ndarray 类型。")

        # 校验 X 和 y 的形状是否匹配
        if X.shape[0] != y.shape[0]:
            logging.error("X 和 y 的样本数量 (行数) 必须一致。")
            raise ValueError("X 和 y 的样本数量 (行数) 必须一致。")

        logging.info("X.shape=%s, y.shape=%s", X.shape, y.shape)

        # 复制 X 和 y，避免操作原始数据
        X_copy = X.copy()
        y_copy = y.copy()
        logging.info("已复制 X 和 y，用于后续操作。")

        # 2. 根据 y 划分 L 和 U
        labeled_mask = ~np.isnan(y_copy)  # 有标签数据的布尔掩码
        unlabeled_mask = np.isnan(y_copy)  # 无标签数据的布尔掩码

        X_L = X_copy[labeled_mask]  # 有标签数据的特征
        y_L = y_copy[labeled_mask]  # 有标签数据的标签

        X_U = X_copy[unlabeled_mask]  # 无标签数据的特征

        logging.info("有标签数据 (L): %d 样本", X_L.shape[0])
        logging.info("无标签数据 (U): %d 样本", X_U.shape[0])

        # 根据有标签数据的情况，判断任务类型
        task_type = determine_task_type(y_L)

        if task_type == "classification":
            logging.info("这是一个分类任务。")
            self.__fit__clf(X_L, y_L, X_U)
        elif task_type == "regression":
            logging.info("这是一个回归任务。")
            self.__fit__reg(X_L, y_L, X_U)
        else:
            logging.warning("无法确定任务类型，默认视为分类任务处理。")
            self.__fit__clf(X_L, y_L, X_U)

        logging.info("TwoStep 模型训练完成。")
        return self

    def __fit__clf(self, X_L, y_L, X_U):
        """
        X_L, y_L: 已标注数据及其标签
        X_U: 未标注数据
        self.k: 取最高置信度样本的比例，例如 10% 则 k=0.1
        self.max_iter: 最大迭代次数
        """
        # 初始化预测结果数组
        self.pred = np.zeros(len(X_U))
        logging.info("开始自训练迭代，共计划进行 %d 次迭代", self.max_iter)

        for epoch in range(self.max_iter):
            logging.info("===== 第 %d 次迭代开始 =====", epoch)
            logging.info("当前 Labeled 数据量: %d；Unlabeled 数据量: %d", len(X_L), len(X_U))

            # 1. 对当前 Labeled 数据进行训练
            logging.info("开始训练分类器...")
            self.clf.fit(X_L, y_L)
            logging.info("分类器训练完成")

            # 2. 对 Unlabeled 数据打分
            scores = self.clf.predict_proba(X_U).max(axis=1)
            logging.info("已完成对 Unlabeled 数据的置信度打分")

            # 3. 选出最高置信度的那部分数据（比例可自定义）
            idx = get_top_k_percent_idx(scores, self.k, pick_lowest=False)  # 例如取前 k=10% 的高置信度样本
            logging.info("本轮选出高置信度样本数量: %d，占比约为 %.2f%%", len(idx), self.k * 100)

            # 4. 得到这部分数据的预测标签
            best_pred = self.clf.predict(X_U)[idx]
            # 把 self.pred 对应位置赋值为 best_pred
            self.pred[idx] = best_pred

            # 5. 将这些样本从 X_U 中移除，并“转正”到 X_L, y_L
            # 5.1 将高置信度样本及其预测标签合并到 Labeled 集
            X_L = np.concatenate([X_L, X_U[idx]], axis=0)
            y_L = np.concatenate([y_L, best_pred], axis=0)
            logging.info("已将高置信度样本合并到 Labeled 集，合并后 Labeled 数据量: %d", len(X_L))

            # 5.2 从 Unlabeled 集中删除这些样本
            X_U = np.delete(X_U, idx, axis=0)
            logging.info("从 Unlabeled 集中删除高置信度样本后，剩余 Unlabeled 数据量: %d", len(X_U))

            # 6. 如果此时 X_U 已经为空，就结束迭代
            if len(X_U) == 0:
                logging.info("在 epoch=%d 时，Unlabeled 数据已空，提前结束迭代", epoch)
                break

            logging.info("===== 第 %d 次迭代结束 =====", epoch)

        logging.info("训练流程结束。共进行了 %d 次迭代，目前尚有 %d 个样本未获得预测标签", epoch, len(X_U))

    def __fit__reg(self, X_L, y_L, X_U):
        pass
