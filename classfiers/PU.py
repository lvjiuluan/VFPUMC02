import numpy as np
import logging
from .Bagging_PU import Bagging_PU

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PU:
    def __init__(self, bagging_pu, config):
        # 断言 bagging_pu 必须是 Bagging_PU 类或其子类的实例
        assert isinstance(bagging_pu, Bagging_PU), "bagging_pu 必须是 Bagging_PU 类或其子类的实例"
        self.bagging_pu = bagging_pu
        self.config = config
        self.iP_find_history = []  # 记录每次迭代找到的正样本
        self.iN_find_history = []  # 记录每次迭代找到的负样本
        self.randomID_history = []  # 记录每次迭代随机选择的样本
        self.iP_new = []  # 记录从未标记中新找到的正样本
        self.iN_new = []  # 记录从未标记样本中新找到的负样样本
        self.pred_history = []  # 记录每次迭代的预测结果
        self.pred = None  # 当前的预测结果
        self.X = None
        self.y = None
        logging.info("PU 类初始化完成")

    def fit(self, X, y):
        logging.info("开始训练 PU 模型")
        self.X = X.copy()
        self.y = y.copy()
        positive_indices = np.where(y == 1)[0]
        negative_indices = np.where(y == 0)[0]
        bagging_pu = self.bagging_pu

        # 初次训练 Bagging_PU
        logging.info("开始初次训练 Bagging_PU 模型")
        bagging_pu.fit(X, y)
        self.pred = bagging_pu.get_out_of_bag_score()
        logging.info("初次训练完成，out-of-bag 预测得分已保存")

        # 迭代训练
        for i in range(self.config['n_iter']):
            logging.info(f"开始第 {i + 1} 次迭代")
            logging.info(f"标签总数量: {len(y)}，正样本标签数量: {sum(y == 1)}, 负样本标签数量: {sum(y == 0)}, 未标记样本数量: {sum(y == -1)}")

            if self.config.get('num_p_in_hidden') is not None:
                if len(self.iP_new) >= self.config['num_p_in_hidden']:
                    logging.info(f"已经把所有隐藏正样本找到，结束迭代")
                    break
            else:
                logging.warning("配置项 'num_p_in_hidden' 为空或不是整数，无法进行判断")

            if sum(y==-1) == 0:
                # 打日志
                logging.info(f"未标记样本数量为0，结束迭代")
                break

            # 获取正样本和负样本
            RP, RN = self.__getPositive__(self.config['theta_P'], self.config['theta_N'], self.pred, y)

            # 处理交集
            RP, RN, has_intersection = self._handle_intersection(RP, RN, i)

            # 训练、更新标签
            self._update_labels(y, RP, RN, i)

            # 保存数据
            self._save_data(self.pred, y, positive_indices, negative_indices, RP, RN, i)

            # 如果有交集，退出循环
            if has_intersection:
                logging.info(f"第 {i + 1} 次迭代中 RP 和 RN 有交集，已退出循环")
                break

    def _handle_intersection(self, RP, RN, iteration):
        """
        检查并处理 RP 和 RN 的交集，返回处理后的 RP 和 RN 以及是否有交集的标志。
        """
        RP_set = set(RP)
        RN_set = set(RN)
        intersection = RP_set & RN_set  # 交集部分

        if intersection:
            logging.info(f"第 {iteration + 1} 次迭代中 RP 和 RN 有交集，交集大小为 {len(intersection)}")

            # 去除交集部分，保留没有交集的部分
            RP_set -= intersection
            RN_set -= intersection

            # 更新 RP 和 RN
            RP = list(RP_set)
            RN = list(RN_set)

            logging.info(f"第 {iteration + 1} 次迭代去除交集后，剩余 {len(RP)} 个正样本, {len(RN)} 个负样本")
            return RP, RN, True  # 返回处理后的 RP, RN 以及有交集的标志

        return RP, RN, False  # 没有交集，返回 False

    def _update_labels(self, y, RP, RN, iteration):
        """
        更新标签，将 RP 对应的标签设为 1，RN 对应的标签设为 0。
        """
        y[RP] = 1
        y[RN] = 0
        logging.info(f"第 {iteration + 1} 次迭代更新了标签")

    def _save_data(self, pred, y, positive_indices, negative_indices, RP, RN, iteration):
        """
        保存当前迭代的数据。
        """
        self.__saveDate__(pred, y, positive_indices, negative_indices, RP, RN)
        logging.info(f"第 {iteration + 1} 次迭代找到 {len(RP)} 个新的正样本, {len(RN)} 个新的负样本")

    def _retrain_bagging_pu(self, X, y, iteration):
        """
        重新训练 Bagging_PU 模型，并更新预测得分。
        """
        self.bagging_pu.fit(X, y)
        self.pred = self.bagging_pu.get_out_of_bag_score()
        logging.info(f"第 {iteration + 1} 次迭代训练完成，out-of-bag 预测得分已更新")

    def __getPositive__(self, theta_P, theta_N, pred, y):
        """
        根据阈值 theta 找到新的正样本 (RP) 和负样本 (RN)
        只对未标记样本 (y == -1) 的 pred 进行排序，取出前 k 个最大值和最小值的索引
        """
        # 计算需要选择的未标记样本数量 k
        k_p = round(len(y[y == -1]) * theta_P)
        k_n = round(len(y[y == -1]) * theta_N)

        # 找到未标记样本的索引
        unlabelled_indices = np.where(y == -1)[0]

        # 针对未标记样本的 pred 进行排序
        unlabelled_pred = pred[unlabelled_indices]

        # 取出前 k 个最大值的索引 (RP)
        top_k_indices_in_unlabelled = unlabelled_pred.argsort()[-k_p:]

        # 取出前 k 个最小值的索引 (RN)
        bottom_k_indices_in_unlabelled = unlabelled_pred.argsort()[:k_n]

        # 将这些索引映射回原始的索引空间
        RP = unlabelled_indices[top_k_indices_in_unlabelled]
        RN = unlabelled_indices[bottom_k_indices_in_unlabelled]

        logging.info(f"根据阈值 {theta_P} 找到 {k_p} 个新的正样本 (RP) 和 {k_n} 个新的负样本 (RN)")
        return RP, RN

    def __saveDate__(self, pred, y, positive_indices, negative_indices, RP, RN):
        """
        保存每次迭代的状态，包括找到的正样本、负样本、随机选择的样本和预测结果
        """
        # 找到当前正样本的索引
        iP = np.where(y == 1)[0]  # 使用 np.where 直接找到正样本索引
        iP_find = np.setdiff1d(iP, positive_indices)  # 使用 np.setdiff1d 计算差集
        self.iP_find_history.append(iP_find)
        logging.info(f"找到 {len(iP_find)} 个新的正样本")

        # 找到当前负样本的索引
        iN = np.where(y == 0)[0]  # 使用 np.where 直接找到负样本索引
        iN_find = np.setdiff1d(iN, negative_indices)  # 使用 np.setdiff1d 计算差集
        self.iN_find_history.append(iN_find)
        logging.info(f"找到 {len(iN_find)} 个新的负样本")

        # 找到当前未标记样本的索引
        iU = np.where(y == -1)[0]  # 使用 np.where 直接找到未标记样本索引
        if len(iU) > len(iP_find):
            randomID = np.random.choice(iU, size=len(iP_find), replace=False)  # 随机选择未标记样本
            self.randomID_history.append(randomID)
            logging.info(f"随机选择了 {len(randomID)} 个样本")

        # 保存当前的预测结果
        self.pred_history.append(pred)
        logging.info(f"保存了当前的预测结果")

        # 更新正样本列表，使用 set 来避免重复
        self.iP_new = list(set(self.iP_new).union(RP))  # 使用 set.union 更新正样本列表
        logging.info(f"更新了新找到的正样本列表，共 {len(self.iP_new)} 个正样本")

        # 更新负样本列表，使用 set 来避免重复
        self.iN_new = list(set(self.iN_new).union(RN))  # 使用 set.union 更新负样本列表
        logging.info(f"更新了新找到的负样本列表，共 {len(self.iN_new)} 个负样本")

    # 新增方法：获取找到的正样本历史
    def get_positive_history(self):
        return self.iP_find_history

    # 新增方法：获取随机选择的样本历史
    def get_random_id_history(self):
        return self.randomID_history

    # 新增方法：获取预测历史
    def get_prediction_history(self):
        return self.pred_history

    # 新增方法：获取新找到的正样本
    def get_new_positives(self):
        return self.iP_new

    import numpy as np

    def get_predicted_labels(self):
        """
        根据正样本索引 iP_new 和标签数组 y，生成一个预测标签数组。
        iP_new 位置为 1，y 中值为 -1 的位置设置为 0。

        参数:
        iP_new (np.ndarray): 正样本的索引数组。
        y (np.ndarray): 原始标签数组，形状任意。

        返回:
        np.ndarray: 形状与 y 相同的预测标签数组，iP_new 位置为 1，y 中值为 -1 的位置为 0。
        """
        # 创建一个与 y 相同的副本，避免直接修改 self.y
        predicted_labels = np.copy(self.y)

        # 将 iP_new 中的索引位置的值设置为 1
        predicted_labels[self.iP_new] = 1

        # 将 y 中值为 -1 的位置设置为 0
        predicted_labels[predicted_labels == -1] = 0

        return predicted_labels

    def get_predict_proba(self):
        return self.pred