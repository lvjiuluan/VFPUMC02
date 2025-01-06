import numpy as np

class SelfTrainingRegressor(object):
    def __init__(self, regressor, max_iter, confidence_threshold=0.1):
        """
        自训练回归器

        参数
        ----------
        regressor : sklearn回归器
            基学习器
        max_iter : 整数
            最大迭代次数
        confidence_threshold : 浮点数
            置信度阈值，用于选择预测值误差小于该阈值的样本
        """
        self.regressor = regressor
        self.max_iter = max_iter
        self.confidence_threshold = confidence_threshold

    def fit(self, X, X_no_label, Y):
        """
        训练模型

        参数
        ----------
        X : ndarray 形状 (n_labeled_samples, n_features)
            有标签的数据
        X_no_label : ndarray 形状 (n_unlabeled_samples, n_features)
            无标签的数据
        Y : ndarray 形状 (n_labeled_samples,)
            标签
        """
        iter_no = 0
        self.final_regressor = self.regressor

        if self.max_iter == 1:
            print('仅启动监督学习', end='...')
        else:
            print('启动自训练回归器 ...\n')
            print(' | '.join([name.center(11) for name in ["迭代次数", "有标签样本数"]]))

        while (X_no_label.shape[0] > 0) & (iter_no < self.max_iter):

            # 在有标签数据上训练回归器
            self.final_regressor = self.final_regressor.fit(X, Y)

            # 获取无标签数据的预测值
            predictions = self.final_regressor.predict(X_no_label)

            # 计算预测值的置信度（这里使用预测值的绝对误差作为置信度）
            confidence = np.abs(predictions - np.mean(predictions))

            # 选择置信度高的样本（误差小于阈值）
            ix = confidence < self.confidence_threshold

            # 更新有标签数据集
            X = np.concatenate([X, X_no_label[ix]], axis=0)
            Y = np.concatenate([Y, predictions[ix]])
            X_no_label = X_no_label[~ix]

            # 增加一次迭代
            iter_no += 1

            # 监控进度
            if self.max_iter == 1:
                print('完成')
            else:
                print(' | '.join([("%d" % iter_no).rjust(11), ("%d" % sum(ix)).rjust(11)]))

    def predict(self, X):
        """
        使用最终的回归器进行预测

        参数
        ----------
        X : ndarray 形状 (n_samples, n_features)

        返回值
        ---------
        predictions: ndarray 形状 (n_samples, )
        """
        return self.final_regressor.predict(X)

    def mean_squared_error(self, Xtest, Ytrue):
        """
        计算均方误差

        参数
        ----------
        Xtest : ndarray 形状 (n_samples, n_features)
            测试数据
        Ytrue : ndarray 形状 (n_samples, )
            测试标签

        返回值
        ---------
        mse: 浮点数
            均方误差
        """
        predictions = self.predict(Xtest)
        mse = np.mean((predictions - Ytrue) ** 2)
        print('均方误差: {}'.format(round(mse, 4)))
        return mse
