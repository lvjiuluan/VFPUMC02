import sys
sys.path.append(r'D:\PyCharmProjects\VFPUMC02')
sys.path.append(r'C:\Users\Administrator\PycharmProjects\VFPUMC02')
sys.path.append(r'/root/VFPUMC02')
%load_ext autoreload
%autoreload 2

from datasets.DataSet import BankDataset
from utils.DataProcessUtils import *
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
from gen.VFPU_GEN import VFPU_GEN
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

dataset = VFPU_GEN_DATASET()

ctgan = CTGAN(epochs=5)

clf =  LogisticRegression(solver='sag', max_iter=5)
reg =  Ridge(alpha=0.1, solver='sag', max_iter=5)
two_step = TwoStep(
    base_classifier=clf,
    base_regressor=reg,
    max_iter=5,
    k=0.25
)

vf_clf = VF_LR(learning_rate=0.2,epoch_num=5,batch_size=64)
vf_reg = VF_LinearRegression(config={'lr': 0.01,'lambda': 0.1,'n_iters':5})
vf_two_step = VF_TwoStep(
    clf=vf_clf,
    reg=vf_reg,
    k=0.25,
    max_iter=5
)

for dataset_enum in DataSetsName:
    print(dataset_enum.value)
    df_A, df_B, y = dataset.get_dataset_parts(dataset_enum.value)
    print(f"df_A 的形状: {df_A.shape}")
    print(f"df_B 的形状: {df_B.shape}")
    print(f"y 的形状: {y.shape}")
    miss_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for miss_rate in miss_rates:
        print(f"miss_rate的值为{miss_rate}")
        complete_df_B, incomplete_df_B, missing_df_B = constru_row_miss_df(df_B, miss_rate)
        print(f"complete_df_B 的形状: {complete_df_B.shape}")
        print(f"incomplete_df_B 的形状: {incomplete_df_B.shape}")
        print(f"missing_df_B 的形状: {missing_df_B.shape}")
        vfpu_gen = VFPU_GEN(two_step,vf_two_step,ctgan,df_A,incomplete_df_B)
        m = incomplete_df_B.shape[1]
        # 生成 k 列， 预测 m - k 列
        for k in range(m + 1):
            predict_cols = incomplete_df_B.columns[:m-k]
            vfpu_gen.fit(predict_cols)
            rmse, mse, mae, r2 = evaluate_imputed_data(missing_df_B, synthetic_data)
            print(f"rmse = {rmse}, mse = {mse}, mae = {mae}, r2 = {r2} ")

