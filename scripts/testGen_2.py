import sys
sys.path.append(r'D:\PyCharmProjects\VFPUMC02')
sys.path.append(r'C:\Users\Administrator\PycharmProjects\VFPUMC02')
sys.path.append(r'/root/VFPUMC02')
#%load_ext autoreload
#%autoreload 2

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
import logging
import pickle
import pandas as pd
from datetime import datetime

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


# 初始化结果字典
result = {}

# 遍历数据集
for dataset_enum in DataSetsName:
    dataset_name = dataset_enum.value
    result[dataset_name] = {}
    logging.info(f"开始处理数据集: {dataset_name}")

    # 获取数据集的部分
    df_A, df_B, y = dataset.get_dataset_parts(dataset_name)
    logging.info(f"df_A 的形状: {df_A.shape}, df_B 的形状: {df_B.shape}, y 的形状: {y.shape}")

    miss_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for miss_rate in miss_rates:
        result[dataset_name][miss_rate] = {}
        logging.info(f"开始处理 miss_rate = {miss_rate}")

        # 构造缺失数据
        complete_df_B, incomplete_df_B, missing_df_B = constru_row_miss_df(df_B, miss_rate)
        logging.info(
            f"complete_df_B 的形状: {complete_df_B.shape}, incomplete_df_B 的形状: {incomplete_df_B.shape}, missing_df_B 的形状: {missing_df_B.shape}")

        vfpu_gen = VFPU_GEN(two_step, vf_two_step, ctgan, df_A, incomplete_df_B)
        m = incomplete_df_B.shape[1]

        # 生成 k 列，预测 m - k 列
        for k in range(m + 1):
            predict_cols = incomplete_df_B.columns[:m - k]
            vfpu_gen.fit(predict_cols)
            synthetic_data = vfpu_gen.get_synthetic_data()

            # 评估生成数据
            rmse, mse, mae, r2 = evaluate_imputed_data(missing_df_B, synthetic_data)
            logging.info(f"miss_rate = {miss_rate}, k = {k}, rmse = {rmse}, mse = {mse}, mae = {mae}, r2 = {r2}")

            # 保存结果
            result[dataset_name][miss_rate][k] = [rmse, mse, mae, r2]

        # 定期保存结果到文件
        with open("result.pkl", "wb") as f:
            pickle.dump(result, f)
        logging.info(f"中间结果已保存到 result.pkl")

# 转换结果为 DataFrame
logging.info("开始将结果转换为 DataFrame")
rows = []
for dataset_name, miss_rate_data in result.items():
    for miss_rate, k_data in miss_rate_data.items():
        for k, metrics in k_data.items():
            rows.append([dataset_name, miss_rate, k] + metrics)

columns = ["Dataset", "MissRate", "K", "RMSE", "MSE", "MAE", "R2"]
result_df = pd.DataFrame(rows, columns=columns)

# 保存结果为 CSV 文件
result_df.to_csv("result.csv", index=False)
logging.info("结果已保存到 result.csv")

