import time

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
        self.pred = None  # 初始化预测结果
        logging.info("TwoStep 类已初始化: max_iter=%d, k=%.2f, base_classifier=%s, base_regressor=%s",
                     max_iter, k, type(base_classifier).__name__, type(base_regressor).__name__)

    def fit(self, X_L, y_L, X_U):
        """
            训练 TwoStep 模型。

            参数:
            - X_L (numpy ndarray): 有标签数据的特征矩阵，形状为 (n_labeled_samples, n_features)。
            - y_L (numpy ndarray): 有标签数据的标签向量，形状为 (n_labeled_samples, )。
            - X_U (numpy ndarray): 无标签数据的特征矩阵，形状为 (n_unlabeled_samples, n_features)。

            返回:
            - self: 训练后的模型。
        """
        logging.info("有标签数据 (L): %d 样本", X_L.shape[0])
        logging.info("无标签数据 (U): %d 样本", X_U.shape[0])

        # 根据有标签数据的情况，判断任务类型
        task_type = determine_task_type(y_L)
        logging.info("任务类型判定结果: %s", task_type)

        if task_type == "classification":
            logging.info("这是一个分类任务。")
            # 记录分类任务中的类别信息
            unique_classes = np.unique(y_L)
            logging.info("分类任务中的类别: %s", unique_classes)
            self.__fit__clf(X_L, y_L, X_U)
        elif task_type == "regression":
            logging.info("这是一个回归任务。")
            # 记录回归任务中的标签统计信息
            y_min, y_max, y_mean, y_std = y_L.min(), y_L.max(), y_L.mean(), y_L.std()
            logging.info("回归任务标签统计: 最小值=%.4f, 最大值=%.4f, 平均值=%.4f, 标准差=%.4f",
                         y_min, y_max, y_mean, y_std)
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
        unlabeled_indices = np.arange(len(X_U))
        self.pred = np.full(len(X_U), np.nan)
        logging.info("初始化预测结果数组，长度=%d", len(self.pred))
        logging.info("开始自训练迭代，共计划进行 %d 次迭代", self.max_iter)
        start_time = time.time()

        for epoch in range(self.max_iter):
            epoch_start_time = time.time()
            logging.info("===== 第 %d 次迭代开始 =====", epoch)
            logging.info("当前 Labeled 数据量: %d；Unlabeled 数据量: %d", len(X_L), len(X_U))

            if len(X_L) == 0:
                logging.warning("当前没有任何有标签的数据，跳过本次迭代。")
                break

            # 1. 对当前 Labeled 数据进行训练
            logging.info("开始训练分类器 (基于 %d 个样本)...", len(X_L))
            clf_start_time = time.time()
            self.clf.fit(X_L, y_L)
            clf_time = time.time() - clf_start_time
            logging.info("分类器训练完成，耗时 %.2f 秒。", clf_time)

            # 记录训练后的分类器的一些信息
            if hasattr(self.clf, 'classes_'):
                logging.info("分类器识别的类别: %s", self.clf.classes_)
            else:
                logging.info("分类器没有 'classes_' 属性。")

            # 2. 对 Unlabeled 数据打分
            logging.info("开始对 Unlabeled 数据进行置信度打分...")
            scores_start_time = time.time()
            proba = self.clf.predict_proba(X_U)
            scores = proba.max(axis=1)
            scores_time = time.time() - scores_start_time
            logging.info("完成置信度打分，耗时 %.2f 秒。", scores_time)
            logging.debug("置信度分数统计: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                          scores.min(), scores.max(), scores.mean(), scores.std())

            # 3. 选出最高置信度的那部分数据（比例可自定义）
            logging.info("选取前 %.2f%% 的高置信度样本。", self.k * 100)
            idx = get_top_k_percent_idx(scores, self.k, pick_lowest=False)
            logging.info("本轮选出高置信度样本数量: %d，占比约为 %.2f%%", len(idx), self.k * 100)
            logging.debug("选中的样本索引: %s", idx)
            selected_original_idx = unlabeled_indices[idx]
            logging.debug("选中的原始样本索引: %s", selected_original_idx)

            # 4. 得到这部分数据的预测标签
            logging.info("对选中的高置信度样本进行预测标签。")
            best_pred = self.clf.predict(X_U)[idx]
            logging.debug("预测标签分布: %s", np.unique(best_pred, return_counts=True))

            # 把 self.pred 对应位置赋值为 best_pred
            self.pred[selected_original_idx] = best_pred
            logging.debug("更新 self.pred 的值。")

            # 5. 将这些样本从 X_U 中移除，并“转正”到 X_L, y_L
            logging.info("将高置信度样本合并到 Labeled 集。")
            # 5.1 将高置信度样本及其预测标签合并到 Labeled 集
            X_L = np.concatenate([X_L, X_U[idx]], axis=0)
            y_L = np.concatenate([y_L, best_pred], axis=0)
            logging.info("已将高置信度样本合并到 Labeled 集，合并后 Labeled 数据量: %d", len(X_L))

            # 5.2 从 Unlabeled 集中删除这些样本
            X_U = np.delete(X_U, idx, axis=0)
            unlabeled_indices = np.delete(unlabeled_indices, idx, axis=0)
            logging.info("从 Unlabeled 集中删除高置信度样本后，剩余 Unlabeled 数据量: %d", len(X_U))

            # 6. 如果此时 X_U 已经为空，就结束迭代
            if len(X_U) == 0:
                logging.info("在 epoch=%d 时，Unlabeled 数据已空，提前结束迭代", epoch)
                break

            epoch_elapsed_time = time.time() - epoch_start_time
            logging.info("===== 第 %d 次迭代结束，耗时 %.2f 秒 =====", epoch, epoch_elapsed_time)

        total_elapsed_time = time.time() - start_time
        logging.info("训练流程结束。共进行了 %d 次迭代，耗时 %.2f 秒。", epoch, total_elapsed_time)
        logging.info("目前尚有 %d 个样本未获得预测标签。", len(X_U))
        if len(X_U) != 0:
            logging.info("还有 %d 个样本没有预测, self.pred中为nan的数量 = %d, len(X_U) = %d, len(unlabeled_indices)"
                         , len(X_U), np.isnan(self.pred).sum(), len(X_U), len(unlabeled_indices))
            # 还有一些未标记样本没有预测，需要预测
            logging.info("对剩余未标记的样本进行最终预测。")
            final_pred_start_time = time.time()
            final_pred = self.clf.predict(X_U)
            final_pred_time = time.time() - final_pred_start_time
            logging.info("完成最终预测，耗时 %.2f 秒。", final_pred_time)
            logging.debug("最终预测标签分布: %s", np.unique(final_pred, return_counts=True))
            # 将最终剩余未标记数据的预测结果，映射回对应的 self.pred 索引
            self.pred[unlabeled_indices] = final_pred

    def __fit__reg(self, X_L, y_L, X_U):
        pass
