from .BaggingPU import Bagging_PU, np


class PU:
    def __init__(self, bagging_pu, config):
        # 断言 bagging_pu 必须是 bagging_pu 类或其子类的实例
        assert isinstance(bagging_pu, Bagging_PU), "bagging_pu 必须是 bagging_pu 类或其子类的实例"
        self.bagging_pu = bagging_pu
        self.config = config
        self.iP_find_history = []
        self.randomID_history = []
        self.iP_new = []
        self.pred_history = []

    def fit(self, X, y):
        ID = X[:, 0].astype(np.int64)
        bagging_pu = self.bagging_pu
        bagging_pu.fit(X, y)
        self.pred = bagging_pu.get_out_of_bag_score()
        for i in range(self.config['n_iter']):
            self.__saveDate__(pred, y, ID, RP)
            print('*' * 7 + "第%d次迭代" % (i + 1) + '*' * 7)
            bagging_pu.fit(X, y)
            pred = bagging_pu.get_out_of_bag_score()
            RP = self.__getPositive__(self.config['theta'], pred, y)
            y[RP] = 1

    def __getPositive__(self, theta, pred, y):
        k = round(len(y[y == 0]) * theta)
        RP = pred.argsort()[-k:]
        return RP

    def __saveDate__(self, pred, y, ID, RP):
        iP = np.array([i for i in range(len(y)) if y[i] == 1])
        iP_find = np.array(list(set(iP) - set(ID)))
        self.iP_find_history.append(iP_find)
        iU = np.array([i for i in range(len(y)) if y[i] == 0])
        randomID = np.random.choice(iU, replace=False, size=len(iP_find))
        self.randomID_history.append(randomID)
        self.pred_history.append(pred)
        for idRP in RP[::-1]:
            if idRP not in self.iP_new:
                self.iP_new.append(idRP)


class Two_Step_PU:
    def __init__(self, base_estimator):
        self.bc = base_estimator

    def fit(self, X, y):
        self.bc.fit(X, y)


class Bagging_PU:
    pass


class Standard_PU:
    pass
