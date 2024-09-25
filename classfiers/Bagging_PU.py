from datetime import time

from utils.DataProcessUtils import *

import numpy as np
import time
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Bagging_PU():
    def __init__(self, base_estimator, config):
        self.base_estimator = base_estimator
        self.config = config
        self.clfSpendTime = []  # 用于记录每次基分类器的训练时间
        self.predictSpendTime = []  # 用于记录每次预测的时间
        self.out_of_bag = None  # 用于保存out-of-bag得分
        logging.info("Bagging_PU 初始化完成")

    def fit(self, X, y):
        logging.info("开始训练 Bagging_PU 模型")
        iP = np.array([i for i in range(len(y)) if y[i] == 1])
        iN = np.array([i for i in range(len(y)) if y[i] == 0])
        diff = len(iP) - len(iN)
        logging.info(f"正样本数量: {len(iP)}, 负样本数量: {len(iN)}, 差值: {diff}")

        if diff >= 0:
            self.handle_imbalance(X, y, iP, iN, is_positive=True)
        else:
            self.handle_imbalance(X, y, iP, iN, is_positive=False)

    def handle_imbalance(self, X, y, iP, iN, is_positive):
        iU = np.array([i for i in range(len(y)) if y[i] == -1])
        diff = abs(len(iP) - len(iN))
        logging.info(f"未标记样本数量: {len(iU)}, 样本差值: {diff}")

        # 如果未标记样本不足以平衡数据集，直接训练
        if diff >= len(iU):
            logging.info("未标记样本不足以平衡数据集，直接训练")
            iL = np.array([i for i in range(len(y)) if y[i] != 0])
            self.base_estimator.fit(X[iL], y[iL])
            self.out_of_bag = self.base_estimator.predict_proba(X)[:,1]
            return

        # 记录每一个数据点out-of-bag的总分数和总次数
        num_oob = np.zeros(y.shape)
        sum_oob = np.zeros(y.shape)

        for i in range(self.config['n_estimators']):
            logging.info(f"开始第 {i + 1} 次迭代")

            # 根据正负样本的不同，抽样未标记样本
            if is_positive:
                iN_sample = np.random.choice(iU, replace=False, size=diff)
                iN_combined = np.concatenate((iN, iN_sample))
                i_oob = list(set(iU) - set(iN_combined))
                X_train = np.concatenate((X[iP], X[iN_combined]), axis=0)
                y_train = np.concatenate((np.ones(len(iP)), np.zeros(len(iN_combined))))
            else:
                iP_sample = np.random.choice(iU, replace=False, size=diff)
                iP_combined = np.concatenate((iP, iP_sample))
                i_oob = list(set(iU) - set(iP_combined))
                X_train = np.concatenate((X[iP_combined], X[iN]), axis=0)
                y_train = np.concatenate((np.ones(len(iP_combined)), np.zeros(len(iN))))

            # 训练基分类器
            clfSpendTimeStart = time.time()
            self.base_estimator.fit(X_train, y_train)
            clfSpendTimeEnd = time.time()
            clfSpendTime = clfSpendTimeEnd - clfSpendTimeStart
            logging.info(f"训练第 {i + 1} 次基分类器耗时: {clfSpendTime:.4f} 秒")
            self.clfSpendTime.append(clfSpendTime)

            # 预测袋外样本的得分
            predictStart = time.time()
            sum_oob[i_oob] += self.base_estimator.predict_proba(X[i_oob])[:, 1]
            predictEnd = time.time()
            predictSpend = predictEnd - predictStart
            logging.info(f"预测第 {i + 1} 次袋外样本耗时: {predictSpend:.4f} 秒, 样本大小: {X[i_oob].shape}")
            self.predictSpendTime.append(predictSpend)
            num_oob[i_oob] += 1

        # 计算out-of-bag得分
        num_oob = np.where(num_oob == 0, 1, num_oob)
        self.out_of_bag = sum_oob / num_oob
        logging.info("完成out-of-bag得分计算")

    def get_out_of_bag_score(self):
        logging.info("获取out-of-bag得分")
        return self.out_of_bag

    def get_training_times(self):
        logging.info("获取基分类器训练时间")
        return self.clfSpendTime

    def get_prediction_times(self):
        logging.info("获取预测时间")
        return self.predictSpendTime
