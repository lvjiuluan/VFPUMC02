import abc


class VF_BASE_CLF(metaclass=abc.ABCMeta):
    """
    抽象基类：分类器基类，要求实现 fit、predict 和 predict_proba 方法。
    """

    @abc.abstractmethod
    def fit(self, XA, XB, y):
        """
        训练纵向联邦分类器。
        参数:
        - XA: 特征集 A。
        - XB: 特征集 B。
        - y: 标签。
        """
        pass

    @abc.abstractmethod
    def predict(self, XA, XB):
        """
        使用纵向联邦分类器进行预测。
        参数:
        - XA: 特征集 A。
        - XB: 特征集 B。
        返回:
        - 分类结果。
        """
        pass

    @abc.abstractmethod
    def predict_proba(self, XA, XB):
        """
        使用分类器预测概率。
        参数:
        - XA: 特征集 A。
        - XB: 特征集 B。
        返回:
        - 每个类别的概率，长度和XA、XB的长度一样
        """
        pass


class VF_BASE_REG(metaclass=abc.ABCMeta):
    """
    抽象基类：回归器基类，要求实现 fit 和 predict 方法。
    """

    @abc.abstractmethod
    def fit(self, XA, XB, y):
        """
        训练回归器。
        参数:
        - XA: 特征集 A。
        - XB: 特征集 B。
        - y: 标签。
        """
        pass

    @abc.abstractmethod
    def predict(self, XA, XB):
        """
        使用回归器进行预测。
        参数:
        - XA: 特征集 A。
        - XB: 特征集 B。
        返回:
        - 回归预测结果。
        """
        pass
