from classfiers.VF_BASE import VF_BASE_CLF
from utils.FateUtils import *
from consts.Constants import *
from utils.pklUtils import *
import subprocess
import sys


class VF_SBT(VF_BASE_CLF):
    def __init__(self, config):
        """
        初始化 VF_SBT 实例，加载并更新配置。
        """
        try:
            sbt_config = load_config(SBT_CONFIGS_PATH)
            sbt_config.update(config)
            self.config = sbt_config
            self.result = None
            save_config(sbt_config, SBT_CONFIGS_PATH)
            self._is_fitted = False  # 更具描述性的变量名
        except Exception as e:
            print(f"初始化时发生错误: {e}")
            raise

    def fit(self, XA, XB, y):
        """
        训练 SBT 模型。
        """
        try:
            A_df, B_df = fate_construct_df(XA, XB, y)
            save_host_guest_dataframes(A_df, B_df, A_host_train_path, B_guest_train_path)
            print("VF_SBT训练结束")
            self._is_fitted = False  # 标记模型需要重新预测
        except Exception as e:
            print(f"训练时发生错误: {e}")
            raise

    def predict(self, XA, XB):
        """
        进行预测并返回分类结果。
        """
        return self._execute_prediction(XA, XB, return_proba=False)

    def predict_proba(self, XA, XB):
        """
        进行预测并返回预测概率,多维数组，从0到C，每一列表示一类的概率
        """
        return self._execute_prediction(XA, XB, return_proba=True)

    def _execute_prediction(self, XA, XB, return_proba=False):
        """
        执行预测的私有方法，减少代码重复。

        :param XA: 主数据集
        :param XB: 客数据集
        :param return_proba: 是否返回预测概率
        :return: 预测结果或预测概率
        """
        if self._is_fitted:
            return self.y_proba if return_proba else self.predict_result

        try:
            A_df, B_df = fate_construct_df(XA, XB)
            save_host_guest_dataframes(A_df, B_df, A_host_test_path, B_guest_test_path)
            self.result = execute_sbt_command(self.config)

            # 提取结果
            self.guest_result = self.result.get('guest', {})
            self.host_result = self.result.get('host', {})

            self.guest_model_dict = self.guest_result.get('model_dict', {})
            self.host_model_dict = self.host_result.get('model_dict', {})
            self.pred_df = self.guest_result.get('pred_df', None)

            if self.pred_df is None:
                raise ValueError("预测数据框 `pred_df` 缺失。")

            self.predict_score = self.pred_df.predict_score
            self.predict_result = self.pred_df.predict_result

            self._is_fitted = True  # 标记模型已进行预测

            self.y_proba = parse_probability_details(self.pred_df.predict_detail)

            return self.y_proba if return_proba else self.predict_result

        except Exception as e:
            print(f"预测时发生错误: {e}")
            raise