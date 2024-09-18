from datetime import time

from utils.DataProcessUtils import *


class Bagging_PU():
    def __init__(self, base_estimator, config):
        self.base_estimator = base_estimator
        self.config = config

    def fit(self, X, y):
        iP = np.array([i for i in range(len(y)) if y[i] == 1])
        iN = np.array([i for i in range(len(y)) if y[i] == 0])
        diff = len(iP) - len(iN)
        if diff >= 0:
            self.handle_imbalance(X, y, iP, iN, is_positive=True)
        else:
            self.handle_imbalance(X, y, iP, iN, is_positive=False)

    def handle_imbalance(self, X, y, iP, iN, is_positive):
        iU = np.array([i for i in range(len(y)) if y[i] == -1])
        diff = abs(len(iP) - len(iN))

        # 如果未标记样本不足以平衡数据集，直接训练
        if diff >= len(iU):
            iL = np.array([i for i in range(len(y)) if y[i] != 0])
            self.base_estimator.fit(X[iL], y[iL])
            self.out_of_bag = self.base_estimator.predict_proba(X)
            return

        # 记录每一个数据点out-of-bag的总分数和总次数
        num_oob = np.zeros(y.shape)
        sum_oob = np.zeros(y.shape)

        for i in range(self.config['n_estimators']):
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
            print("训练一次基分类器要%f秒" % clfSpendTime)
            self.clfSpendTime.append(clfSpendTime)

            # 预测袋外样本的得分
            predictStart = time.time()
            sum_oob[i_oob] += self.base_estimator.predict_proba(X[i_oob], X[i_oob])
            predictEnd = time.time()
            predictSpend = predictEnd - predictStart
            print("预测一次大小为(%d,%d)的数据要%f秒" % (X[i_oob].shape[0], X[i_oob].shape[1], predictSpend))
            num_oob[i_oob] += 1

        # 计算out-of-bag得分
        num_oob = np.where(num_oob == 0, 1, num_oob)
        self.out_of_bag = sum_oob / num_oob

    def get_out_of_bag_score(self):
        return self.out_of_bag
