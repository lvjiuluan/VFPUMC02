import sys
sys.path.append('D:\PyCharmProjects\VFPUMC02')
sys.path.append(r'C:\Users\Administrator\PycharmProjects\VFPUMC02')
sys.path.append(r'/root/VFPUMC02')
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline

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
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge


import logging
import time
import pickle
import pandas as pd

# 设置日志格式和级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 用于保存所有中间结果和最终结果的字典
results = {}

# 加载数据集
logging.info("Loading dataset...")
dataset = VFPU_GEN_DATASET()
df_A, df_B, y = dataset.get_dataset_parts(DataSetsName.BANK.value)

# 保存原始数据
results['df_A'] = df_A
results['df_B'] = df_B
results['y'] = y

# 设置缺失率并生成缺失数据
miss_rate = 0.2
logging.info(f"Generating incomplete dataset with missing rate: {miss_rate}")
complete_df_B, incomplete_df_B, missing_df_B = constru_row_miss_df(df_B, miss_rate)

# 保存缺失数据相关结果
results['complete_df_B'] = complete_df_B
results['incomplete_df_B'] = incomplete_df_B
results['missing_df_B'] = missing_df_B

# 初始化CTGAN模型并训练
logging.info("Initializing CTGAN model...")
ctgan = CTGAN(epochs=10)
discrete_columns = get_discrete_columns(incomplete_df_B)
logging.info("Fitting CTGAN model on incomplete data...")
ctgan.fit(incomplete_df_B, discrete_columns)

# 保存CTGAN模型和离散列信息
results['ctgan_model'] = ctgan
results['discrete_columns'] = discrete_columns

# 生成与缺失数据相同数量的合成数据
length = missing_df_B.shape[0]
logging.info(f"Generating synthetic data of length: {length}")
synthetic_data = ctgan.sample(length)

# 保存生成的合成数据
results['synthetic_data'] = synthetic_data

# 构建完整的df_B
logging.info("Constructing complete df_B with synthetic data...")
construct_df_B = stack_and_reset_index(incomplete_df_B, synthetic_data)

# 保存构建的完整数据
results['construct_df_B'] = construct_df_B

# 评估填补的合成数据
logging.info("Evaluating imputed data...")
imputed_eval_ctgan = evaluate_imputed_data(missing_df_B.values, synthetic_data.values)
results['imputed_eval_ctgan'] = imputed_eval_ctgan

# 获取未标记行索引和预测列
unlabeled_row_indices = list(missing_df_B.index)
predict_cols = construct_df_B.columns

# 处理数据框架
logging.info("Processing dataframes for labeled and unlabeled data...")
df_A_L, df_A_U, construct_df_B_L, construct_df_B_U, y_L_dict, y_U_dict, construct_df_B_L_train, construct_df_B_U_train = process_dataframes(
    df_A, construct_df_B, unlabeled_row_indices, predict_cols
)

# 初始化分类器和回归器
logging.info("Initializing Logistic Regression and Ridge models...")
clf = LogisticRegression(solver='sag', max_iter=100)
reg = Ridge(alpha=0.1, solver='sag', max_iter=100)

# 初始化TwoStep模型
logging.info("Initializing TwoStep model...")
two_step = TwoStep(
    base_classifier=clf,
    base_regressor=reg,
    max_iter=100,
    k=0.1
)

# 使用TwoStep模型预测未标记数据
y_pred_dict = {}
logging.info("Starting TwoStep model training and prediction...")
start_time = time.time()
for key, y_L in y_L_dict.items():
    logging.info(f"Processing label: {key}")
    two_step.fit(df_A_L.values, y_L.values, df_A_U.values)
    y_pred = two_step.get_unlabeled_predict_by_label(y_L)
    y_pred_dict[key] = y_pred
logging.info(f"TwoStep model completed in {time.time() - start_time:.2f} seconds.")

# 保存TwoStep模型预测结果
results['y_pred_dict_two_step'] = y_pred_dict

# 将预测结果转换为DataFrame
synthetic_data_gen_all = pd.DataFrame(y_pred_dict)
synthetic_data_gen_all = synthetic_data_gen_all[incomplete_df_B.columns]

# 保存TwoStep生成的合成数据
results['synthetic_data_gen_all'] = synthetic_data_gen_all

# 再次评估填补的合成数据
logging.info("Evaluating imputed data with TwoStep predictions...")
imputed_eval_two_step = evaluate_imputed_data(missing_df_B.values, synthetic_data_gen_all)
results['imputed_eval_two_step'] = imputed_eval_two_step

# 设置新的预测列范围
m = incomplete_df_B.shape[1]
k = 1
unlabeled_row_indices = list(missing_df_B.index)
predict_cols = construct_df_B.columns[0:m - k]

# 处理数据框架
logging.info("Processing dataframes for VF_TwoStep model...")
df_A_L, df_A_U, construct_df_B_L, construct_df_B_U, y_L_dict, y_U_dict, construct_df_B_L_train, construct_df_B_U_train = process_dataframes(
    df_A, construct_df_B, unlabeled_row_indices, predict_cols
)

# 保存处理后的数据（用于VF_TwoStep）
results['predict_cols'] = predict_cols

# 初始化VF_TwoStep模型
logging.info("Initializing VF_TwoStep model...")
vf_clf = VF_LR(
    learning_rate=0.2,
    epoch_num=100,
    batch_size=64
)
vf_reg = VF_LinearRegression(
    config={
        'lr': 0.01,
        'lambda': 0.1,
        'n_iters': 100
    }
)
vf_two_step = VF_TwoStep(
    clf=vf_clf,
    reg=vf_reg,
    k=0.1,
    max_iter=100
)

# 使用VF_TwoStep模型预测未标记数据
y_pred_dict = {}
logging.info("Starting VF_TwoStep model training and prediction...")
start_time = time.time()
for key, y_L in y_L_dict.items():
    logging.info(f"Processing label: {key}")
    vf_two_step.fit(
        XA_L=df_A_L.values,
        XB_L=construct_df_B_L_train.values,
        y_L=y_L,
        XA_U=df_A_U.values,
        XB_U=construct_df_B_U_train.values
    )
    y_pred = vf_two_step.get_unlabeled_predict_by_label(y_L)
    y_pred_dict[key] = y_pred
logging.info(f"VF_TwoStep model completed in {time.time() - start_time:.2f} seconds.")

# 保存VF_TwoStep模型预测结果
results['y_pred_dict_vf_two_step'] = y_pred_dict

# 保存所有结果到gen.pkl文件
logging.info("Saving all results to gen.pkl...")
with open('gen.pkl', 'wb') as f:
    pickle.dump(results, f)

# 日志完成信息
logging.info("Script execution completed. All results saved to gen.pkl.")



