import time

import numpy as np

from enums.ModelType import ModelType
from utils.DataProcessUtils import determine_task_type
from utils.DataProcessUtils import get_top_k_percent_idx
from .VF_BASE import VF_BASE_CLF, VF_BASE_REG
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VF_TwoStep:

    def __init__(self, clf, reg, k=0.1, max_iter=10, min_confidence=0, convergence_threshold=0):
        """
        初始化 VF_TwoStep 类。

        参数:
        ----------
        clf : VF_BASE_CLF 的子类实例
            用于分类任务的分类器，必须是 VF_BASE_CLF 的子类。
        reg : VF_BASE_REG 的子类实例
            用于回归任务的回归器，必须是 VF_BASE_REG 的子类。
        k : float, default=0.1
            每次迭代选出置信度最高的 k% 未标注数据进行自训练。
            取值范围为 (0, 1]。
        max_iter : int, default=10
            自训练的最大迭代次数，必须为正整数。
        min_confidence : float, default=0.0
            用于筛选样本的最低置信度阈值，必须为非负数。
        convergence_threshold : int, default=1
            当本轮选出的高置信度样本数不足该阈值时，提前停止迭代。
            必须为正整数。

        异常:
        ----------
        ValueError:
            当参数不符合要求时抛出 ValueError。
        """
        # 校验 clf 是否为 VF_BASE_CLF 的子类实例
        if not isinstance(clf, VF_BASE_CLF):
            raise ValueError(
                f"clf 必须是 VF_BASE_CLF 的子类实例，但接收到的类型是 {type(clf).__name__}。"
            )

        # 校验 reg 是否为 VF_BASE_REG 的子类实例
        if not isinstance(reg, VF_BASE_REG):
            raise ValueError(
                f"reg 必须是 VF_BASE_REG 的子类实例，但接收到的类型是 {type(reg).__name__}。"
            )

        # 校验 k 是否在 (0, 1] 范围内
        if not (0 < k <= 1):
            raise ValueError(
                f"k 必须在 (0, 1] 范围内，但接收到的值是 {k}。"
            )

        # 校验 max_iter 是否为正整数
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError(
                f"max_iter 必须是正整数，但接收到的值是 {max_iter}。"
            )

        # 校验 min_confidence 是否为非负数
        # if not isinstance(min_confidence, (int, float)) or min_confidence < 0:
        #     raise ValueError(
        #         f"min_confidence 必须是非负数，但接收到的值是 {min_confidence}。"
        #     )

        # 校验 convergence_threshold 是否为正整数
        # if not isinstance(convergence_threshold, int) or convergence_threshold <= 0:
        #     raise ValueError(
        #         f"convergence_threshold 必须是正整数，但接收到的值是 {convergence_threshold}。"
        #     )

        # 初始化参数
        self.clf = clf
        self.reg = reg
        self.k = k
        self.max_iter = max_iter
        self.min_confidence = min_confidence
        self.convergence_threshold = convergence_threshold
        self.pred_clf = None  # 初始化预测结果
        self.pred_reg = None  # 初始化预测结果

        # 打印初始化日志
        logging.info(
            "VF_TwoStep 类已成功初始化: max_iter=%d, k=%.2f, min_confidence=%.2f, convergence_threshold=%d, clf=%s, reg=%s",
            max_iter, k, min_confidence, convergence_threshold, type(clf).__name__, type(reg).__name__
        )

    def fit(self, XA_L, XB_L, y_L, XA_U, XB_U):
        """
        训练 TwoStep 模型。

        参数:
        - XA_L (numpy ndarray): 有标签数据的特征矩阵 A，形状为 (n_labeled_samples, n_features_A)。
        - XB_L (numpy ndarray): 有标签数据的特征矩阵 B，形状为 (n_labeled_samples, n_features_B)。
        - y_L (numpy ndarray): 有标签数据的标签向量，形状为 (n_labeled_samples, )。
        - XA_U (numpy ndarray): 无标签数据的特征矩阵 A，形状为 (n_unlabeled_samples, n_features_A)。
        - XB_U (numpy ndarray): 无标签数据的特征矩阵 B，形状为 (n_unlabeled_samples, n_features_B)。

        返回:
        - self: 训练后的模型实例。
        """
        # 日志打印有标签和无标签数据的样本数量
        logging.info("有标签数据 (L): %d 样本", XA_L.shape[0])
        logging.info("无标签数据 (U): %d 样本", XA_U.shape[0])

        # 根据有标签数据的标签，判断任务类型（分类或回归）
        task_type = determine_task_type(y_L)
        logging.info("任务类型判定结果: %s", task_type)

        # 根据任务类型调用相应的训练方法
        if task_type == "classification":
            logging.info("这是一个分类任务。")
            # 记录分类任务中的类别信息
            unique_classes = np.unique(y_L)
            logging.info("分类任务中的类别: %s", unique_classes)
            self.__fit__clf(XA_L, XB_L, y_L, XA_U, XB_U)
        elif task_type == "regression":
            logging.info("这是一个回归任务。")
            # 记录回归任务中的标签统计信息
            y_min, y_max, y_mean, y_std = y_L.min(), y_L.max(), y_L.mean(), y_L.std()
            logging.info(
                "回归任务标签统计: 最小值=%.4f, 最大值=%.4f, 平均值=%.4f, 标准差=%.4f",
                y_min, y_max, y_mean, y_std
            )
            self.__fit__reg(XA_L, XB_L, y_L, XA_U, XB_U)
        else:
            # 如果无法确定任务类型，默认按分类任务处理
            logging.warning("无法确定任务类型，默认视为分类任务处理。")
            self.__fit__clf(XA_L, XB_L, y_L, XA_U, XB_U)

        logging.info("VF_TwoStep 模型训练完成。")
        return self

    def __fit__clf(self, XA_L, XB_L, y_L, XA_U, XB_U):
        """
        自训练过程的主函数。

        参数:
        ----------
        XA_L : np.ndarray
            已标注数据的 A 方特征，形状为 (n_L, dimA)。
        XB_L : np.ndarray
            已标注数据的 B 方特征，形状为 (n_L, dimB)。
        y_L : np.ndarray
            已标注数据的标签，形状为 (n_L,)。
        XA_U : np.ndarray
            未标注数据的 A 方特征，形状为 (n_U, dimA)。
        XB_U : np.ndarray
            未标注数据的 B 方特征，形状为 (n_U, dimB)。

        返回:
        ----------
        None
            函数执行结束后，会将对未标注数据的预测结果写入 self.pred。
        """
        # 初始化预测结果数组, 与 XA_U, XB_U 行数一致
        unlabeled_indices = np.arange(len(XA_U))
        self.pred_clf = np.full(len(XA_U), np.nan)

        logging.info("[INIT] 初始化 self.pred, 未标注数据数量=%d", len(XA_U))
        logging.info("[INIT] 最大迭代次数 max_iter=%d, 每轮选取比例 k=%.2f", self.max_iter, self.k)

        start_time = time.time()
        for epoch in range(self.max_iter):
            epoch_start_time = time.time()

            logging.info("===== [Epoch %d/%d] 迭代开始 =====", epoch + 1, self.max_iter)
            logging.info("[INFO] 当前 Labeled 样本数量: %d, Unlabeled 样本数量: %d",
                         len(XA_L), len(XA_U))

            # 若当前没有任何有标签数据，无法进行训练，直接结束
            if len(XA_L) == 0:
                logging.warning("[WARNING] Labeled 数据量为0，无法继续训练，终止迭代。")
                break

            # 1. 训练分类器
            logging.info("[STEP 1] 开始训练分类器 (基于 %d 个 Labeled 样本)...", len(XA_L))
            clf_start_time = time.time()
            self.clf.fit(XA_L, XB_L, y_L)
            clf_time = time.time() - clf_start_time
            logging.info("[STEP 1] 分类器训练完成，耗时 %.2f 秒。", clf_time)

            # 打印一下分类器识别到的类别信息(若有)
            if hasattr(self.clf, 'classes_'):
                logging.debug("[DEBUG] 分类器识别的类别列表: %s", self.clf.classes_)
            else:
                logging.debug("[DEBUG] 分类器无 'classes_' 属性，无法输出类别列表。")

            # 2. 对 Unlabeled 数据打分
            logging.info("[STEP 2] 对 Unlabeled 数据进行预测概率计算...")
            scores_start_time = time.time()
            proba = self.clf.predict_proba(XA_U, XB_U)
            # 取预测概率中最大的值作为该样本的置信度
            scores = proba.max(axis=1)
            scores_time = time.time() - scores_start_time
            logging.info("[STEP 2] 预测概率计算完成，耗时 %.2f 秒。", scores_time)
            logging.debug("[DEBUG] 置信度分数统计: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                          scores.min(), scores.max(), scores.mean(), scores.std())

            # 3. 选出最高置信度的那部分数据
            logging.info("[STEP 3] 按置信度选取前 %.2f%% 的未标注样本 (min_confidence=%.2f)...",
                         self.k * 100, self.min_confidence)
            idx = get_top_k_percent_idx(scores, self.k, pick_lowest=False)

            if len(idx) == 0:
                logging.info("[INFO] 本轮没有样本满足置信度筛选条件，终止迭代。")
                break

            logging.info("[STEP 3] 本轮选中高置信度样本数量: %d, 占未标注样本总数的 %.2f%%",
                         len(idx), 100.0 * len(idx) / (len(XA_U) + 1e-9))
            selected_original_idx = unlabeled_indices[idx]

            # 4. 得到这部分数据的预测标签
            logging.info("[STEP 4] 获取高置信度样本的预测标签。")
            best_pred = self.clf.predict(XA_U, XB_U)[idx]

            # 将 self.pred 对应位置赋值为 best_pred
            self.pred_clf[selected_original_idx] = best_pred

            # 5. 将这些样本移动到 Labeled 集合中
            logging.info("[STEP 5] 合并高置信度样本至 Labeled 集...")
            XA_L = np.concatenate([XA_L, XA_U[idx]], axis=0)
            XB_L = np.concatenate([XB_L, XB_U[idx]], axis=0)
            y_L = np.concatenate([y_L, best_pred], axis=0)
            logging.info("[STEP 5] 合并后 Labeled 样本数=%d", len(XA_L))

            # 从 Unlabeled 集中删除这些样本
            XA_U = np.delete(XA_U, idx, axis=0)
            XB_U = np.delete(XB_U, idx, axis=0)
            unlabeled_indices = np.delete(unlabeled_indices, idx, axis=0)
            logging.info("[STEP 5] 剩余 Unlabeled 样本数=%d", len(XA_U))

            # 判断是否满足提前收敛条件（本轮选样本不足某个数量）
            if len(idx) < self.convergence_threshold:
                logging.info("[INFO] 本轮新增标注数量=%d < 收敛阈值=%d，提前结束迭代。",
                             len(idx), self.convergence_threshold)
                break

            # 6. 如果 Unlabeled 集已空，结束迭代
            if len(XA_U) == 0:
                logging.info("[INFO] Unlabeled 样本已全部处理完，提前结束迭代。")
                break

            epoch_elapsed_time = time.time() - epoch_start_time
            logging.info("===== [Epoch %d/%d] 迭代结束，耗时 %.2f 秒 =====",
                         epoch + 1, self.max_iter, epoch_elapsed_time)

        total_elapsed_time = time.time() - start_time
        logging.info("[DONE] 自训练流程结束，共进行了 %d 轮迭代，耗时 %.2f 秒。", epoch + 1, total_elapsed_time)
        logging.info("[DONE] 目前尚有 %d 个未标注样本未获得最终预测。", len(XA_U))

        # 若还有剩余的未标注数据，则进行一次最终预测
        if len(XA_U) != 0:
            logging.info("[FINAL] 对剩余 %d 个未标记样本进行最终预测...", len(XA_U))
            final_pred_start_time = time.time()
            final_pred = self.clf.predict(XA_U, XB_U)
            final_pred_time = time.time() - final_pred_start_time
            logging.info("[FINAL] 最终预测完成，耗时 %.2f 秒。", final_pred_time)
            logging.debug("[DEBUG] 剩余未标注样本的预测标签分布: %s",
                          np.unique(final_pred, return_counts=True))

            # 映射到 self.pred
            self.pred_clf[unlabeled_indices] = final_pred

        logging.info("[DONE] 所有未标注样本的预测任务完成！(self.pred 已更新)")

    def __fit__reg(self, XA_L, XB_L, y_L, XA_U, XB_U):
        """
        自训练过程的主函数。

        参数:
        ----------
        XA_L : np.ndarray
            已标注数据的 A 方特征，形状为 (n_L, dimA)。
        XB_L : np.ndarray
            已标注数据的 B 方特征，形状为 (n_L, dimB)。
        y_L : np.ndarray
            已标注数据的标签，形状为 (n_L,)。
        XA_U : np.ndarray
            未标注数据的 A 方特征，形状为 (n_U, dimA)。
        XB_U : np.ndarray
            未标注数据的 B 方特征，形状为 (n_U, dimB)。

        返回:
        ----------
        None
            函数执行结束后，会将对未标注数据的预测结果写入 self.pred。
        """
        # 初始化预测结果数组, 与 XA_U, XB_U 行数一致
        unlabeled_indices = np.arange(len(XA_U))
        self.pred_reg = np.full(len(XA_U), np.nan)

        logging.info("[INIT] 初始化 self.pred, 未标注数据数量=%d", len(XA_U))
        logging.info("[INIT] 最大迭代次数 max_iter=%d, 每轮选取比例 k=%.2f", self.max_iter, self.k)

        start_time = time.time()
        for epoch in range(self.max_iter):
            epoch_start_time = time.time()

            logging.info("===== [Epoch %d/%d] 迭代开始 =====", epoch + 1, self.max_iter)
            logging.info("[INFO] 当前 Labeled 样本数量: %d, Unlabeled 样本数量: %d",
                         len(XA_L), len(XA_U))

            # 若当前没有任何有标签数据，无法进行训练，直接结束
            if len(XA_L) == 0:
                logging.warning("[WARNING] Labeled 数据量为0，无法继续训练，终止迭代。")
                break

            # 1. 训练回归器
            logging.info("[STEP 1] 开始训练回归器 (基于 %d 个 Labeled 样本)...", len(XA_L))
            reg_start_time = time.time()
            self.reg.fit(XA_L, XB_L, y_L)
            clf_time = time.time() - reg_start_time
            logging.info("[STEP 1] 回归器训练完成，耗时 %.2f 秒。", clf_time)

            # 2. 对 Unlabeled 数据打分
            logging.info("[STEP 2] 对 Unlabeled 数据进行置信度打分...")
            scores_start_time = time.time()
            predictions = self.reg.predict(XA_U, XB_U)
            mean_prediction = np.mean(predictions)
            std_prediction = np.std(predictions)
            # 计算每个预测值的 Z 分数（标准化偏离程度）
            scores = np.abs((predictions - mean_prediction) / std_prediction)
            scores_time = time.time() - scores_start_time
            logging.info("完成置信度打分，耗时 %.2f 秒。", scores_time)
            logging.debug("预测值统计: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                          predictions.min(), predictions.max(), predictions.mean(), predictions.std())
            logging.debug("置信度分数统计: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                          scores.min(), scores.max(), scores.mean(), scores.std())

            # 3. 选出最高置信度的那部分数据
            logging.info("[STEP 3] 按置信度选取前 %.2f%% 的未标注样本 (min_confidence=%.2f)...",
                         self.k * 100, self.min_confidence)
            idx = get_top_k_percent_idx(scores, self.k, pick_lowest=True)  # pick_lowest=True 表示选择偏离均值最小的样本

            if len(idx) == 0:
                logging.info("[INFO] 本轮没有样本满足置信度筛选条件，终止迭代。")
                break

            logging.info("[STEP 3] 本轮选中高置信度样本数量: %d, 占未标注样本总数的 %.2f%%",
                         len(idx), 100.0 * len(idx) / (len(XA_U) + 1e-9))
            selected_original_idx = unlabeled_indices[idx]

            # 4. 得到这部分数据的预测标签
            logging.info("[STEP 4] 获取高置信度样本的预测标签。")
            best_pred = predictions[idx]

            # 将 self.pred 对应位置赋值为 best_pred
            self.pred_reg[selected_original_idx] = best_pred
            logging.debug("更新 self.pred 的值。")

            # 5. 将这些样本移动到 Labeled 集合中
            logging.info("[STEP 5] 合并高置信度样本至 Labeled 集...")
            XA_L = np.concatenate([XA_L, XA_U[idx]], axis=0)
            XB_L = np.concatenate([XB_L, XB_U[idx]], axis=0)
            y_L = np.concatenate([y_L, best_pred], axis=0)
            logging.info("[STEP 5] 合并后 Labeled 样本数=%d", len(XA_L))

            # 从 Unlabeled 集中删除这些样本
            XA_U = np.delete(XA_U, idx, axis=0)
            XB_U = np.delete(XB_U, idx, axis=0)
            unlabeled_indices = np.delete(unlabeled_indices, idx, axis=0)
            logging.info("[STEP 5] 剩余 Unlabeled 样本数=%d", len(XA_U))

            # 判断是否满足提前收敛条件（本轮选样本不足某个数量）
            if len(idx) < self.convergence_threshold:
                logging.info("[INFO] 本轮新增标注数量=%d < 收敛阈值=%d，提前结束迭代。",
                             len(idx), self.convergence_threshold)
                break

            # 6. 如果 Unlabeled 集已空，结束迭代
            if len(XA_U) == 0:
                logging.info("[INFO] Unlabeled 样本已全部处理完，提前结束迭代。")
                break

            epoch_elapsed_time = time.time() - epoch_start_time
            logging.info("===== [Epoch %d/%d] 迭代结束，耗时 %.2f 秒 =====",
                         epoch + 1, self.max_iter, epoch_elapsed_time)

        total_elapsed_time = time.time() - start_time
        logging.info("[DONE] 自训练流程结束，共进行了 %d 轮迭代，耗时 %.2f 秒。", epoch + 1, total_elapsed_time)
        logging.info("[DONE] 目前尚有 %d 个未标注样本未获得最终预测。", len(XA_U))

        # 若还有剩余的未标注数据，则进行一次最终预测
        if len(XA_U) != 0:
            logging.info("[FINAL] 对剩余 %d 个未标记样本进行最终预测...", len(XA_U))
            final_pred_start_time = time.time()
            final_pred = self.reg.predict(XA_U, XB_U)
            final_pred_time = time.time() - final_pred_start_time
            logging.info("[FINAL] 最终预测完成，耗时 %.2f 秒。", final_pred_time)
            logging.debug("[DEBUG] 剩余未标注样本的预测标签分布: %s",
                          np.unique(final_pred, return_counts=True))

            # 映射到 self.pred
            self.pred_reg[unlabeled_indices] = final_pred

        logging.info("[DONE] 所有未标注样本的预测任务完成！(self.pred 已更新)")

    def get_unlabeled_predict(self, model_type=ModelType.CLASSIFICATION):
        if model_type == ModelType.CLASSIFICATION:
            # 修改为int类型，只针对预测类型
            self.pred_clf = self.pred_clf.astype('int')
            return self.pred_clf
        elif model_type == ModelType.REGRESSION:
            return self.pred_reg
        else:
            raise ValueError("Invalid model type specified")

