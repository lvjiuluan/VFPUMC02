import sys
sys.path.append('D:\PyCharmProjects\VFPUMC02')
sys.path.append(r'C:\Users\Administrator\PycharmProjects\VFPUMC02')
sys.path.append(r'/root/VFPUMC02')
%load_ext autoreload
%autoreload 2

%matplotlib inline

from datasets.DataSet import BankDataset
from utils.DataProcessUtils import (
    convert_columns_to_int,
    shuffle_column_order,
    vertical_split,
    constru_row_miss_df,
    get_discrete_columns,
    evaluate_imputed_data,
    find_rounds_math,
    stack_and_reset_index,
    process_dataframes
)
from consts.Constants import DATASETS_PATH
import os
from datasets.VFPU_GEN_DATASET import VFPU_GEN_DATASET
from enums.DataSetsName import DataSetsName
from ctgan import CTGAN
from classfiers.TwoStep import TwoStep
import pandas as pd
from vf4lr.VF_LR import VF_LR
from classfiers.VF_LinearRegression import VF_LinearRegression
from classfiers.VF_TwoStep import VF_TwoStep
from utils.FateUtils import convert_ipynb_to_py

dataset = VFPU_GEN_DATASET()

df_A, df_B, y = dataset.get_dataset_parts(DataSetsName.BANK.value)

miss_rate=0.2
complete_df_B, incomplete_df_B, missing_df_B = constru_row_miss_df(df_B, miss_rate)

ctgan = CTGAN(epochs=10)
discrete_columns = get_discrete_columns(incomplete_df_B)
ctgan.fit(incomplete_df_B,discrete_columns)
length=missing_df_B.shape[0]
# 生成数据
synthetic_data = ctgan.sample(length)

construct_df_B = stack_and_reset_index(incomplete_df_B, synthetic_data)

evaluate_imputed_data(missing_df_B.values, synthetic_data.values)

unlabeled_row_indices = list(missing_df_B.index)
predict_cols = construct_df_B.columns

df_A_L, df_A_U, construct_df_B_L, construct_df_B_U, y_L_dict, y_U_dict, construct_df_B_L_train, construct_df_B_U_train = process_dataframes(df_A, construct_df_B, unlabeled_row_indices, predict_cols)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
clf =  LogisticRegression(solver='sag', max_iter=100)
reg =  Ridge(alpha=0.1, solver='sag', max_iter=100)

two_step = TwoStep(
    base_classifier=clf,
    base_regressor=reg,
    max_iter=100,
    k=0.1
)

y_pred_dict = {}
for key, y_L in y_L_dict.items():
    two_step.fit(df_A_L.values, y_L.values, df_A_U.values)
    y_pred = two_step.get_unlabeled_predict_by_label(y_L)
    y_pred_dict[key] = y_pred

synthetic_data_gen_all = pd.DataFrame(y_pred_dict)

synthetic_data_gen_all = synthetic_data_gen_all[incomplete_df_B.columns]

synthetic_data_gen_all.shape

missing_df_B.shape

evaluate_imputed_data(missing_df_B.values, synthetic_data_gen_all)

m = incomplete_df_B.shape[1]
for k in range(m+1):
    predict_cols = construct_df_B.columns[0:m-k]
    print(predict_cols)

k = 1
unlabeled_row_indices = list(missing_df_B.index)
predict_cols = construct_df_B.columns[0:m-k]

df_A_L, df_A_U, construct_df_B_L, construct_df_B_U, y_L_dict, y_U_dict, construct_df_B_L_train, construct_df_B_U_train = process_dataframes(df_A, construct_df_B, unlabeled_row_indices, predict_cols)

vf_clf = VF_LR(
    learning_rate=0.2,
    epoch_num=100,
    batch_size=64
)
vf_reg = VF_LinearRegression(
    config={
    'lr': 0.01,
    'lambda': 0.1,
    'n_iters':100
    }
)

vf_two_step = VF_TwoStep(
    clf=vf_clf,
    reg=vf_reg,
    k=0.1,
    max_iter=100
)

y_pred_dict = {}
for key, y_L in y_L_dict.items():
    vf_two_step.fit(
    XA_L=df_A_L.values,
    XB_L=construct_df_B_L_train.values,
    y_L=y_L,
    XA_U=df_A_U.values,
    XB_U=construct_df_B_U_train.values
    )
    y_pred = vf_two_step.get_unlabeled_predict_by_label(y_L)
    y_pred_dict[key] = y_pred

convert_ipynb_to_py('测试生成.ipynb','testGen_1')

