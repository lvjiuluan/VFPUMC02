from abc import ABC, abstractmethod

# 定义 Base_Estimator 类
class VF_Base_Estimator(ABC):
    @abstractmethod
    def fit(self, XA, XB, y):
        """子类必须实现 fit 方法"""
        pass

    @abstractmethod
    def predict_proba(self, XA, XB):
        """子类必须实现 predict_proba 方法，并返回与 XA 长度相同的 ndarray"""
        pass