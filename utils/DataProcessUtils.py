import os
from sklearn.semi_supervised import LabelSpreading, LabelPropagation, SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import yaml

from consts.Constants import CONFITS_PATH
from datasets.DataSet import DataSet
from enums.HideRatio import HideRatio


def evaluate_model(y_true, y_pred, y_prob):
    """
    评估模型的准确率、召回率、AUC 和 F1 分数。
    如果输入包含 NaN 值，则将其替换为 0。
    """
    y_true = np.nan_to_num(y_true, nan=0)
    y_pred = np.nan_to_num(y_pred, nan=0)
    y_prob = np.nan_to_num(y_prob, nan=0)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)

    return accuracy, recall, auc, f1


def semi_supervised_evaluation(dataset):
    """
    对传入的 dataset 进行半监督学习评估，使用 Label Spreading、Label Propagation 和 Self-training 方法。
    结果存储在一个 DataFrame 中，行标签为 HideRatio 的成员名，列标签为方法和评估指标的组合。

    参数:
    - dataset: 包含 df, hidden_y 和 y 属性的对象。

    返回:
    - results_df: 包含评估结果的 DataFrame。
    """
    assert isinstance(dataset, DataSet), "dataset 必须是 DataSet 类或其子类的实例"
    # 创建一个空的 DataFrame，用于存储结果
    methods = ['Label Spreading', 'Label Propagation', 'Self-training']
    metrics = ['Accuracy', 'Recall', 'AUC', 'F1']
    index = pd.MultiIndex.from_product([methods, metrics], names=['Method', 'Metric'])
    results_df = pd.DataFrame(columns=index)

    # 遍历 HideRatio 枚举类
    for member in HideRatio:
        print(f"当前的 member 为 {member.name}")

        # 获取隐藏标签
        dataset.get_hidden_labels(member)

        # 提取数据
        X = dataset.df
        hidden_y = dataset.hidden_y
        y_true = dataset.y

        # 找到未标记样本的索引
        unlabeled_mask = hidden_y == -1

        # 找到已标记样本的索引
        labeled_mask = hidden_y != -1

        # 只对未标记样本进行预测，因此我们需要对这些样本的预测结果进行评估
        X_unlabeled = X[unlabeled_mask]
        y_true_unlabeled = y_true[unlabeled_mask]  # 用于评估的真实标签

        # 对数据进行拆分，用于 Label Propagation 和 Self-training
        X_small, _, hidden_y_small, _ = train_test_split(X, hidden_y, test_size=0.9, random_state=42)

        # 1. Label Spreading
        label_spread = LabelSpreading(kernel='knn', n_neighbors=5)
        label_spread.fit(X, hidden_y)

        # 对未标记样本进行预测
        y_pred_spread = label_spread.predict(X_unlabeled)
        y_prob_spread = label_spread.predict_proba(X_unlabeled)[:, 1]

        # 评估 Label Spreading
        accuracy_spread, recall_spread, auc_spread, f1_spread = evaluate_model(y_true_unlabeled, y_pred_spread,
                                                                               y_prob_spread)

        # 2. Label Propagation
        label_prop = LabelPropagation(kernel='rbf', gamma=20)
        label_prop.fit(X_small, hidden_y_small)

        # 对未标记样本进行预测
        y_pred_prop = label_prop.predict(X_unlabeled)
        y_prob_prop = label_prop.predict_proba(X_unlabeled)[:, 1]

        # 评估 Label Propagation
        accuracy_prop, recall_prop, auc_prop, f1_prop = evaluate_model(y_true_unlabeled, y_pred_prop, y_prob_prop)

        # 3. Self-training
        svc = SVC(probability=True)
        self_training_model = SelfTrainingClassifier(base_estimator=svc, threshold=0.5)
        self_training_model.fit(X_small, hidden_y_small)

        # 对未标记样本进行预测
        y_pred_self_train = self_training_model.predict(X_unlabeled)
        y_prob_self_train = self_training_model.predict_proba(X_unlabeled)[:, 1]

        # 评估 Self-training
        accuracy_self_train, recall_self_train, auc_self_train, f1_self_train = evaluate_model(y_true_unlabeled,
                                                                                               y_pred_self_train,
                                                                                               y_prob_self_train)

        # 将结果存储到 DataFrame 中
        results_df.loc[member.name, ('Label Spreading', 'Accuracy')] = accuracy_spread
        results_df.loc[member.name, ('Label Spreading', 'Recall')] = recall_spread
        results_df.loc[member.name, ('Label Spreading', 'AUC')] = auc_spread
        results_df.loc[member.name, ('Label Spreading', 'F1')] = f1_spread

        results_df.loc[member.name, ('Label Propagation', 'Accuracy')] = accuracy_prop
        results_df.loc[member.name, ('Label Propagation', 'Recall')] = recall_prop
        results_df.loc[member.name, ('Label Propagation', 'AUC')] = auc_prop
        results_df.loc[member.name, ('Label Propagation', 'F1')] = f1_prop

        results_df.loc[member.name, ('Self-training', 'Accuracy')] = accuracy_self_train
        results_df.loc[member.name, ('Self-training', 'Recall')] = recall_self_train
        results_df.loc[member.name, ('Self-training', 'AUC')] = auc_self_train
        results_df.loc[member.name, ('Self-training', 'F1')] = f1_self_train

    return results_df



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
