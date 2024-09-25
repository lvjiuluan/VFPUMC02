import os

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score
import numpy as np
import yaml
from consts.Constants import CONFITS_PATH


def subtract_from_row(df: pd.DataFrame, row_no: int, diff: float) -> pd.DataFrame:
    """
    将指定行的每个值减去 diff，并返回修改后的 DataFrame。

    参数:
    - df: 需要操作的 DataFrame。
    - row_no: 需要操作的行号（从 0 开始）。
    - diff: 需要从该行的每个值中减去的数值。

    返回:
    - 修改后的 DataFrame。
    """
    # 检查行号是否在 DataFrame 的范围内
    if row_no < 0 or row_no >= len(df):
        raise IndexError("行号超出 DataFrame 的范围")

    # 将指定行的每个值减去 diff
    df.loc[row_no] = df.loc[row_no] - diff

    return df


def value_counts_for_labels(*ys):
    """
    统计多个标签数组中每个标签值的出现次数。

    参数:
    *ys: 可变数量的标签数组（np.ndarray），每个数组代表一个标签集合。

    返回:
    None: 直接打印每个标签数组中不同值的出现次数。
    """
    for idx, y in enumerate(ys):
        print(f"标签数组 {idx + 1}:")
        # 使用 np.unique 统计不同值的个数
        unique_values, counts = np.unique(y, return_counts=True)

        # 打印结果
        for value, count in zip(unique_values, counts):
            print(f"  值 {value} 出现了 {count} 次")
        print()  # 每个数组之间空一行


def evaluate_model(y_true, y_pred, y_prob):
    """
    评估模型的准确率、召回率、AUC 和 F1 分数。
    如果输入包含 NaN 值，则将其替换为 0。

    参数:
    y_true (np.ndarray): 真实标签的数组，通常是二分类问题中的 0 或 1。
    y_pred (np.ndarray): 模型预测的标签数组，通常是二分类问题中的 0 或 1。
    y_prob (np.ndarray): 模型预测的概率数组，表示每个样本属于正类（1）的概率。

    返回:
    tuple: 包含以下四个评估指标的元组：
        - accuracy (float): 准确率，表示预测正确的样本占总样本的比例。
        - recall (float): 召回率，表示在所有正类样本中被正确预测为正类的比例。
        - auc (float): AUC（ROC 曲线下面积），表示模型区分正负类的能力。
        - f1 (float): F1 分数，精确率和召回率的调和平均数。
    """
    y_true = np.nan_to_num(y_true, nan=0)
    y_pred = np.nan_to_num(y_pred, nan=0)
    y_prob = np.nan_to_num(y_prob, nan=0)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)

    return accuracy, recall, auc, f1





def validate_input(XA, XB, y):
    """
    验证 XA, XB, y 是否为 numpy 的 ndarray 类型，并且长度相同。

    :param XA: numpy ndarray
    :param XB: numpy ndarray
    :param y: numpy ndarray
    :raises AssertionError: 如果输入不符合要求
    """
    # 断言判断 XA, XB, y 为 numpy 的 ndarray 类型
    assert isinstance(XA, np.ndarray), "XA 必须是 numpy 的 ndarray 类型"
    assert isinstance(XB, np.ndarray), "XB 必须是 numpy 的 ndarray 类型"
    assert isinstance(y, np.ndarray), "y 必须是 numpy 的 ndarray 类型"

    # 断言判断 XA, XB, y 的长度相同
    assert len(XA) == len(XB) == len(y), "XA, XB 和 y 的长度必须相同"


def getConfigYaml(configName):
    configFileName = f"{configName}.yaml"
    configFilePath = os.path.join(CONFITS_PATH, configFileName)
    # 读取 YAML 文件
    with open(configFilePath, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def vertical_split(originalDf, split_rate):
    # 确保输入参数正确
    if abs(sum(split_rate) - 1) > 1e-6:
        raise ValueError("split_rate的和必须为1")

    # 复制DataFrame以避免修改原始数据
    df = originalDf.copy()

    # 打乱除了'y'之外的所有列
    cols_to_shuffle = df.columns.difference(['y'])
    df[cols_to_shuffle] = df[cols_to_shuffle].sample(frac=1).reset_index(drop=True)

    # 计算垂直切分的列数
    total_cols = len(cols_to_shuffle)
    split_col_index = int(total_cols * split_rate[0])

    # 根据split_rate切分DataFrame的列
    cols_df1 = cols_to_shuffle[:split_col_index]
    cols_df2 = cols_to_shuffle[split_col_index:]
    df1 = df[cols_df1]
    df2 = df[cols_df2]

    return df1, df2


def split_and_hide_labels(originalDf, split_rate, unlabeled_rate):
    # 确保输入参数正确
    if abs(sum(split_rate) - 1) > 1e-6:
        raise ValueError("split_rate的和必须为1")
    if not (0 <= unlabeled_rate < 1):
        raise ValueError("unlabeled_rate必须在[0, 1)范围内")

    # 复制DataFrame以避免修改原始数据
    df = originalDf.copy()

    # 打乱除了'y'之外的所有列
    cols_to_shuffle = df.columns.difference(['y'])
    df[cols_to_shuffle] = df[cols_to_shuffle].sample(frac=1).reset_index(drop=True)

    # 计算垂直切分的列数
    total_cols = len(cols_to_shuffle)
    split_col_index = int(total_cols * split_rate[0])

    # 根据split_rate切分DataFrame的列
    cols_df1 = cols_to_shuffle[:split_col_index]
    cols_df2 = cols_to_shuffle[split_col_index:]
    df1 = df[cols_df1]
    df2 = df[cols_df2]

    # 随机选择标签置为-1
    num_labels_to_hide = int(len(df) * unlabeled_rate)
    indices_to_hide = np.random.choice(df.index, num_labels_to_hide, replace=False)
    y_modified = df['y'].copy()
    y_modified.loc[indices_to_hide] = -1

    return df1, df2, y_modified, df['y']


def print_column_types(df):
    categorical_count = 0
    numerical_count = 0

    # 遍历DataFrame中的每一列
    for column in df.columns:
        unique_values = df[column].unique()

        # 检查唯一值是否只包含0.0和1.0
        if set(unique_values).issubset({0.0, 1.0}):
            categorical_count += 1
        else:
            numerical_count += 1

    # 直接打印结果
    print("分类列的数量:", categorical_count)
    print("数值列的数量:", numerical_count)
