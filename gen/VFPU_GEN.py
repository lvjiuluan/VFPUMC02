from classfiers.TwoStep import TwoStep
from classfiers.VF_TwoStep import VF_TwoStep
from utils.DataProcessUtils import get_unlabeled_row_indices, stack_and_reset_index, process_dataframes
import pandas as pd


class VFPU_GEN:
    def __init__(self, two_step, vf_two_step, ctgan):
        # 校验two_step和vf_two_step必须分别为TwoStep类和VF_TwoStep的子类
        self.two_step = two_step
        self.vf_two_step = vf_two_step
        self.ctgan = ctgan

        self.complete_df_A = None
        self.incomplete_df_B = None
        self.unlabeled_row_indices = None
        self.synthetic_data = None
        self.construct_df_B = None

    def fit(self, complete_df_A, incomplete_df_B, predict_cols):
        self.complete_df_A = complete_df_A
        self.incomplete_df_B = incomplete_df_B
        # 获取未对齐的部分行索引
        self.unlabeled_row_indices = get_unlabeled_row_indices(complete_df_A, incomplete_df_B)
        self.ctgan.fit(incomplete_df_B, incomplete_df_B.columns)
        self.synthetic_data = self.ctgan.sample(len(complete_df_A) - len(incomplete_df_B))
        self.construct_df_B = stack_and_reset_index(incomplete_df_B, self.synthetic_data)
        if predict_cols is None or len(predict_cols) == 0:
            print("训练完成")
            return

        df_A_L, df_A_U, construct_df_B_L, construct_df_B_U, y_L_dict, y_U_dict, \
        construct_df_B_L_train, construct_df_B_U_train = process_dataframes(
            self.complete_df_A,
            self.construct_df_B,
            self.unlabeled_row_indices,
            predict_cols
        )
        if construct_df_B_L_train is None:
            y_pred_dict = {}
            for key, y_L in y_L_dict.items():
                self.two_step.fit(df_A_L.values, y_L.values, df_A_U.values)
                y_pred = self.two_step.get_unlabeled_predict_by_label(y_L)
                y_pred_dict[key] = y_pred
            imputed_data = pd.DataFrame(y_pred_dict)


    def get_synthetic_data(self):
        return self.synthetic_data