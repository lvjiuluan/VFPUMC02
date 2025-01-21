import abc

class VF_BASE(metaclass=abc.ABCMeta):
    """
    抽象基类：纵向联邦学习模型基类，要求实现 fit 和 predict 方法。
    """

    @abc.abstractmethod
    def fit(self, XA, XB, y):
        """
        训练模型。
        参数:
        - XA: 特征集 A。
        - XB: 特征集 B。
        - y: 标签。
        """
        pass

    @abc.abstractmethod
    def predict(self, XA, XB):
        """
        使用模型进行预测。
        参数:
        - XA: 特征集 A。
        - XB: 特征集 B。
        返回:
        - 预测结果。
        """
        pass


class VF_BASE_CLF(VF_BASE):
    """
    抽象基类：分类器基类，继承自 VF_BASE，要求额外实现 predict_proba 方法。
    """

    @abc.abstractmethod
    def predict_proba(self, XA, XB):
        """
        使用分类器预测概率。
        参数:
        - XA: 特征集 A。
        - XB: 特征集 B。
        返回:
        - 每个类别的概率，长度和 XA、XB 的长度一样。
        """
        pass


class VF_BASE_REG(VF_BASE):
    """
    抽象基类：回归器基类，继承自 VF_BASE。
    """
    pass
