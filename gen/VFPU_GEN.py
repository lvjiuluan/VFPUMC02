from classfiers.TwoStep import TwoStep
from classfiers.VF_TwoStep import VF_TwoStep
from utils.DataProcessUtils import get_unlabeled_row_indices, stack_and_reset_index, process_dataframes, \
    update_dataframe_with_dict
import pandas as pd

from utils.Logger import Logger

import pandas as pd


class VFPU_GEN:
    """
    VFPU_GEN 类用于基于完整数据集 A 和不完整数据集 B，结合 CTGAN 生成的合成数据进行插补。
    主要功能包括：
      - 生成合成数据并与原始不完整数据 B 结合
      - 根据传入的 two_step 或 vf_two_step 模型对未标记的数据进行预测
      - 最终输出插补后的结果

    参数
    ----------
    two_step : TwoStep
        用于在全部列预测情形下进行预测的模型，须为 TwoStep 类及其子类实例
    vf_two_step : VF_TwoStep
        用于仅预测部分列时进行插补的模型，须为 VF_TwoStep 类及其子类实例
    ctgan : object
        用于生成合成数据的 CTGAN 或类似模型实例，需包含 fit 和 sample 方法
    complete_df_A : pd.DataFrame
        完整数据集 A
    incomplete_df_B : pd.DataFrame
        不完整数据集 B
    logger_level : str, optional
        日志级别，默认为 'ERROR'

    属性
    ----------
    synthetic_data : pd.DataFrame
        由 ctgan 生成的合成数据
    construct_df_B : pd.DataFrame
        将不完整数据 B 与合成数据拼接后得到的结果
    imputed_data : pd.DataFrame
        最终插补/预测后的结果
    """

    def __init__(
            self,
            two_step,
            vf_two_step,
            ctgan,
            complete_df_A,
            incomplete_df_B,
            logger_level='ERROR'
    ):
        """
        初始化 VFPU_GEN 对象，并生成合成数据。

        参数
        ----------
        two_step : TwoStep
            用于在全部列预测情形下的模型，须为 TwoStep 类(或其子类)实例
        vf_two_step : VF_TwoStep
            用于仅预测部分列时的模型，须为 VF_TwoStep 类(或其子类)实例
        ctgan : object
            用于生成合成数据的 CTGAN 或者类似模型实例
        complete_df_A : pd.DataFrame
            完整数据集 A
        incomplete_df_B : pd.DataFrame
            不完整数据集 B
        logger_level : str
            日志级别，如 'INFO', 'DEBUG', 'ERROR'，默认 'ERROR'
        """

        # 在此可添加对传入模型类型的检查，如 isinstance(two_step, TwoStep)，视需求而定
        self.two_step = two_step
        self.vf_two_step = vf_two_step
        self.ctgan = ctgan

        self.complete_df_A = complete_df_A
        self.incomplete_df_B = incomplete_df_B

        # 获得 DataFrame 行列数，方便后续使用
        self.n, self.m = self.incomplete_df_B.shape

        # 获取未对齐（未标记）的行索引信息
        self.unlabeled_row_indices = get_unlabeled_row_indices(complete_df_A, incomplete_df_B)

        # 日志器
        self.logger = Logger.create_new_logger(Logger.getLevelName(logger_level))

        # 拟合 CTGAN 模型以生成合成数据
        # 注意检查待 sample 的条数是否为正数
        sample_count = len(self.complete_df_A) - len(self.incomplete_df_B)
        if sample_count <= 0:
            self.logger.warning(f"完整数据集 A 的长度 <= 不完整数据集 B 的长度，"
                                f"sample_count={sample_count}，将不会生成额外合成数据。")
            self.synthetic_data = pd.DataFrame()
        else:
            self.logger.debug("正在使用 CTGAN 拟合并生成合成数据...")
            self.ctgan.fit(self.incomplete_df_B, self.incomplete_df_B.columns)
            self.synthetic_data = self.ctgan.sample(sample_count)

        # 拼接不完整数据 B 与生成的合成数据
        self.construct_df_B = stack_and_reset_index(incomplete_df_B, self.synthetic_data)
        self.logger.debug("拼接完成，construct_df_B 形状: "
                          f"{self.construct_df_B.shape}")

        # 结果 DataFrame 初始化
        self.imputed_data = pd.DataFrame()

    def fit(self, predict_cols=None):
        """
        根据给定的预测列进行训练/插补。
        若 predict_cols 未指定或为空，则直接返回合成数据。

        参数
        ----------
        predict_cols : list, optional
            待插补预测的列名列表。如果为空或 None，则直接返回合成数据，不做插补。

        返回
        ----------
        None
            结果将保存在 self.imputed_data 属性中。
        """
        if not predict_cols:  # None or len(predict_cols) == 0
            self.logger.info("predict_cols 为空，直接返回合成数据。")
            self.imputed_data = self.synthetic_data
            self.logger.info("训练完成（无额外插补操作）。")
            return

        self.logger.debug(f"开始处理数据集，准备插补列：{predict_cols}")
        (
            df_A_L,  # 已标记的 A
            df_A_U,  # 未标记的 A
            construct_df_B_L,  # 已标记的 B
            construct_df_B_U,  # 未标记的 B
            y_L_dict,  # 已标记的 Y
            y_U_dict,  # 未标记的 Y
            construct_df_B_L_train,
            construct_df_B_U_train
        ) = process_dataframes(
            self.complete_df_A,
            self.construct_df_B,
            self.unlabeled_row_indices,
            predict_cols
        )

        # 如果待预测列等于 B 的全部列，使用 two_step 完成预测
        if len(predict_cols) == self.m:
            self.logger.debug("检测到需要预测所有列，使用 two_step 进行插补...")
            y_pred_dict = {}
            for key, y_L in y_L_dict.items():
                self.two_step.fit(df_A_L.values, y_L.values, df_A_U.values)
                y_pred = self.two_step.get_unlabeled_predict_by_label(y_L)
                y_pred_dict[key] = y_pred

            # 将预测结果转化为 DataFrame
            self.imputed_data = pd.DataFrame(y_pred_dict)
            self.logger.info("训练完成（全部列插补）。")
            return

        # 否则使用 vf_two_step 模型预测指定列
        self.logger.debug("开始使用 vf_two_step 进行部分列插补...")
        y_pred_dict = {}
        for key, y_L in y_L_dict.items():
            self.logger.debug(f"插补列：{key}")
            self.vf_two_step.fit(
                XA_L=df_A_L.values,
                XB_L=construct_df_B_L_train.values,
                y_L=y_L,
                XA_U=df_A_U.values,
                XB_U=construct_df_B_U_train.values
            )
            y_pred = self.vf_two_step.get_unlabeled_predict_by_label(y_L)
            y_pred_dict[key] = y_pred

        # 将预测结果合并至原数据
        self.imputed_data = update_dataframe_with_dict(construct_df_B_U, y_pred_dict)
        self.logger.info("训练完成（部分列插补）。")

    def get_synthetic_data(self):
        """
        获取插补/预测后的数据。如果在调用 fit 前调用该方法，则返回值可能为空。

        Returns
        -------
        pd.DataFrame
            返回插补或合成后的最终结果
        """
        return self.imputed_data
