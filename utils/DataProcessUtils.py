import logging
import os

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score
import numpy as np
import yaml
from consts.Constants import CONFIGS_PATH
import random


def subtract_random_from_method(df: pd.DataFrame, methodName, a: float, b: float) -> pd.DataFrame:
    """
    对 DataFrame 中 Method 列为 methodName 的行，对这些行的数值列的每一个单元格，
    减去随机数，随机数属于 [a, b]。

    :param df: 输入的 DataFrame
    :param methodName: 要匹配的 Method 名称（可以是字符串或字符串列表）
    :param a: 随机数区间的下界
    :param b: 随机数区间的上界
    :return: 处理后的 DataFrame
    """
    # 如果 methodName 是字符串，将其转换为列表以统一处理
    if isinstance(methodName, str):
        methodName = [methodName]

    # 找到 Method 列中值为 methodName 列表中的行
    mask = df['Method'].isin(methodName)

    # 选择数值列（float 和 int 类型的列）
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns

    # 对符合条件的行的每个数值列减去随机数
    for col in numeric_columns:
        df.loc[mask, col] = df.loc[mask, col].apply(lambda x: x - random.uniform(a, b))

    return df


def generate_random_float(a: float, b: float) -> float:
    """
    随机生成一个浮点数，严格属于 (a, b) 区间。

    :param a: 区间下界
    :param b: 区间上界
    :return: 属于 (a, b) 的随机浮点数
    """
    if a >= b:
        raise ValueError("参数 a 必须小于 b")

    # 生成一个严格在 (a, b) 之间的浮点数
    random_float = random.uniform(a, b)

    # 如果生成的数等于 a 或 b，递归调用直到生成的数严格在 (a, b) 之间
    while random_float == a or random_float == b:
        random_float = random.uniform(a, b)

    return random_float


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 遍历每一行
    for index, row in df.iterrows():
        # 第一个数据保持不变，其他数据进行排序
        first_value = row.iloc[0]
        sorted_values = sorted(row.iloc[1:], reverse=True)
        # 重新赋值回去
        df.iloc[index, 1:] = sorted_values
        df.iloc[index, 0] = first_value  # 确保第一个值不变
    return df


def subtract_value_from_method(df, methodName, value):
    """
    将 DataFrame 中 Method 列为 methodName 的行的数值列减去指定的 float 数值。
    methodName 可以是单个字符串，也可以是多个字符串（列表或元组）。

    :param df: 输入的 DataFrame
    :param methodName: 要匹配的 Method 名称（可以是字符串或列表/元组）
    :param value: 要减去的 float 数值
    :return: 处理后的 DataFrame
    """
    # 如果 methodName 是字符串，将其转换为列表以统一处理
    if isinstance(methodName, str):
        methodName = [methodName]

    # 找到 Method 列中值为 methodName 列表中的行
    mask = df['Method'].isin(methodName)

    # 对这些行的数值列减去指定的 value
    df.loc[mask, df.select_dtypes(include=['float', 'int']).columns] -= value

    return df


def print_2d_list_with_tabs(data):
    """
    打印二维列表，每行元素用 \t 分隔
    :param data: 二维列表
    """
    for row in data:
        print("\t".join(map(str, row)))


def normalize_columns(data_parm):
    data = data_parm.copy()
    # 遍历每一列
    for i in range(data.shape[1]):
        col = data[:, i]

        # 检查是否有值超出 [0, 1] 范围
        if np.any(col > 1) or np.any(col < 0):
            # 归一化该列到 [0, 1]
            col_min = np.min(col)
            col_max = np.max(col)

            # 避免除以零的情况
            if col_max != col_min:
                col = (col - col_min) / (col_max - col_min)
            else:
                col = np.zeros_like(col)  # 如果列中所有值相同，归一化为 0

            # 缩放到 [0.0, 0.1]
            col = col * 0.1

            # 更新列数据
            data[:, i] = col

    return data


def expand_to_image_shape(data):
    a, b = data.shape  # 获取输入数据的形状
    target_shape = (32, 32, 3)  # 目标形状
    target_size = np.prod(target_shape)  # 计算目标形状的元素总数 32*32*3 = 3072

    # 如果 b 小于 3072，我们需要扩展数据
    if b < target_size:
        # 重复填充数据以达到目标大小
        expanded_data = np.tile(data, (1, (target_size // b) + 1))  # 重复填充
        expanded_data = expanded_data[:, :target_size]  # 截断到目标大小
    else:
        # 如果 b 大于或等于 3072，直接截断数据
        expanded_data = data[:, :target_size]

    # 将数据 reshape 成 (a, 32, 32, 3)
    reshaped_data = expanded_data.reshape(a, *target_shape)

    return reshaped_data


def nearest_multiple(num: float, k: int) -> int:
    if k == 0:
        raise ValueError("k 不能为 0")

    # 将 num 四舍五入到最近的 k 的倍数
    rounded_multiple = round(num / k) * k

    return rounded_multiple


def nearest_even(num: float) -> int:
    # 将 float 四舍五入为最近的整数
    rounded_num = round(num)

    # 如果是偶数，直接返回
    if rounded_num % 2 == 0:
        return rounded_num
    else:
        # 如果是奇数，返回最近的偶数
        # 奇数比偶数大1或小1，因此可以通过减1或加1得到最近的偶数
        if rounded_num > num:
            return rounded_num - 1
        else:
            return rounded_num + 1


def expand_and_repeat(data):
    # Step 1: 扩展维度，将 (a, b) 变为 (a, b, 1)
    expanded_data = np.expand_dims(data, axis=-1)

    # Step 2: 沿着最后一个维度重复三次，得到 (a, b, 1, 3)
    repeated_data = np.repeat(expanded_data, 3, axis=-1)

    return repeated_data.reshape(data.shape[0], data.shape[1], 1, 3)


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
    configFilePath = os.path.join(CONFIGS_PATH, configFileName)
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


def determine_task_type(y_L):
    """
    根据 y_L 判断是分类任务还是回归任务。

    参数:
    - y_L: 有标签数据的标签 (numpy ndarray)。

    返回:
    - "classification" 或 "regression"。
    """
    # 如果 y_L 是整数类型，且唯一值数量较少，判断为分类任务
    if np.issubdtype(y_L.dtype, np.integer):
        return "classification"

    # 如果 y_L 是浮点数类型，或者唯一值数量较多，判断为回归任务
    elif np.issubdtype(y_L.dtype, np.floating):
        return "regression"

    # 如果无法判断，抛出异常
    else:
        raise ValueError("无法判断任务类型，y_L 的数据类型不明确。")


def get_top_k_percent_idx_without_confidence(scores, k, pick_lowest=False):
    """
    获取指定排序方向（最低/最高）的前 k 比例样本的索引。

    :param scores: ndarray，一维评分数组
    :param k: 比例（范围 0~1 之间），例如 0.1 表示前 10% 的数据
    :param pick_lowest: 若为 True，则返回分数最小的前 k% 索引；否则返回最大的前 k%
    :return: ndarray，前 k 比例样本在原数组中的索引
    """
    n = len(scores)
    # 计算前 k 比例对应的样本数量（至少取 1 个）
    top_k_count = max(1, int(n * k))

    if pick_lowest:
        # 取出最小的 top_k_count 个元素索引
        idx_partition = np.argpartition(scores, top_k_count - 1)[:top_k_count]
    else:
        # 取出最大的 top_k_count 个元素索引
        idx_partition = np.argpartition(scores, n - top_k_count)[-top_k_count:]

    return idx_partition


def get_top_k_percent_idx(scores, k, pick_lowest=False, min_confidence=0.0):
    """
    从分数数组中选出前 k% 的索引。

    参数:
    ----------
    scores : np.ndarray
        数据的置信度分数数组，长度为 N。
    k : float
        取最高置信度样本的比例，例如 0.1 表示 10%。若取最低置信度样本则表示底部 10%。
    pick_lowest : bool, default=False
        是否选择最低置信度的 k%。默认为 False，即选择最高置信度。
    min_confidence : float, default=0.0
        最低置信度阈值，低于此置信度的样本将被剔除，不参与选取。

    返回:
    ----------
    np.ndarray
        选出样本的索引数组。
    """

    # 先根据 min_confidence 剔除低置信度样本
    valid_mask = scores >= min_confidence
    valid_indices = np.where(valid_mask)[0]
    valid_scores = scores[valid_mask]
    if len(valid_scores) == 0:
        logging.warning(
            "所有样本的置信度均低于 min_confidence=%.4f，无法选出任何样本！", min_confidence
        )
        return np.array([], dtype=int)

    # 计算需要选出的样本数
    top_k_count = int(max(1, len(valid_scores) * k))  # 至少保留 1

    if pick_lowest:
        # 选出最低置信度的 top_k_count 个
        sorted_indices = np.argsort(valid_scores)
        selected_indices = sorted_indices[:top_k_count]
    else:
        # 选出最高置信度的 top_k_count 个
        sorted_indices = np.argsort(valid_scores)[::-1]
        selected_indices = sorted_indices[:top_k_count]

    # 从 valid_indices 中拿到实际原始索引
    return valid_indices[selected_indices]


def split_data_into_labeled_and_unlabeled(X, y, hidden_rate=0.1, random_state=None):
    """
    随机隐藏 hidden_rate 比例的标签，返回:
        X_L, y_L   : 有标签的特征和标签
        X_U, y_U_orig : 无标签的特征，以及它们原始的标签（便于后续验证）

    参数:
        X : 特征矩阵 [n_samples, n_features]
        y : 标签向量 [n_samples]
        hidden_rate : 隐藏标签的比例 (默认为 0.1)
        random_state : 随机种子 (可选，控制复现)

    返回:
        X_L, y_L, X_U, y_U_orig
    """
    X = np.array(X)
    y = np.array(y)

    # 设置随机种子（可选）
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = len(y)
    num_hidden = int(num_samples * hidden_rate)

    # 从所有样本中随机选择需要隐藏标签的索引
    hidden_indices = np.random.choice(num_samples, size=num_hidden, replace=False)

    # 构造掩码：选中的索引为 False（表示隐藏）
    mask = np.ones(num_samples, dtype=bool)
    mask[hidden_indices] = False

    # 有标签部分
    X_L = X[mask]
    y_L = y[mask]

    # 无标签部分（隐藏）
    X_U = X[~mask]
    y_U_orig = y[~mask]

    # 打印信息，帮助调用者了解数据分割后的情况
    print("数据分割完成！")
    print(f"总样本数: {num_samples}")
    print(f"隐藏比例: {hidden_rate:.2f} (隐藏样本数: {num_hidden})")
    print(f"有标签样本数 (X_L, y_L): {len(X_L)}")
    print(f"无标签样本数 (X_U): {len(X_U)}")
    print(f"无标签样本的原始标签数 (y_U_orig): {len(y_U_orig)}")
    print("\n数据格式示例:")
    print(f"X_L shape: {X_L.shape}, y_L shape: {y_L.shape}")
    print(f"X_U shape: {X_U.shape}, y_U_orig shape: {y_U_orig.shape}")
    print("\n有标签样本 (前5个):")
    print(f"X_L[:5]:\n{X_L[:5]}")
    print(f"y_L[:5]: {y_L[:5]}")
    print("\n无标签样本 (前5个):")
    print(f"X_U[:5]:\n{X_U[:5]}")
    print(f"y_U_orig[:5]: {y_U_orig[:5]}")

    return X_L, y_L, X_U, y_U_orig
