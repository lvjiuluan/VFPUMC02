from sklearn.semi_supervised import LabelSpreading, LabelPropagation, SelfTrainingClassifier
from sklearn.svm import SVC
import pandas as pd

from classfiers.PU import PU
from enums.HideRatio import HideRatio
from utils.DataProcessUtils import evaluate_model
from configs.Config import PUConfig
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from classfiers.Bagging_PU import Bagging_PU

from datasets.DataSet import get_all_dataset
from consts.Constants import DATASETS_PATH
import os

# 以下方法不对外暴露：

def evaluate_pu_train_datasets(datasets):
    """
    评估多个数据集和不同隐藏比例下的模型表现，返回包含评估指标的 DataFrame。

    参数:
    datasets (list): 包含多个数据集对象的列表。
    HideRatio (list): 包含不同隐藏比例的列表。
    PUConfig (dict): PU 学习的配置字典。

    返回:
    pd.DataFrame: 包含评估结果的 DataFrame，一级索引为 dataset.baseFileName，二级索引为 member.name。
    """
    # 用于存储结果的列表
    results = []

    # 遍历每个数据集
    for dataset in datasets:
        # 遍历每个隐藏比例
        for member in HideRatio:
            # 获取隐藏标签
            dataset.get_hidden_labels(member)
            # 更新 PUConfig 中的隐藏正样本数量
            PUConfig['num_p_in_hidden'] = dataset.count_hidden_positives()
            # 打印数据集信息
            dataset.datasetInfo()

            # 初始化 GBDT 分类器
            gbdt_classifier = GradientBoostingClassifier()

            # 初始化 Bagging_PU 模型
            bagging_pu = Bagging_PU(base_estimator=gbdt_classifier, config=PUConfig)

            # 训练 Bagging_PU 模型
            bagging_pu.fit(dataset.df.values, dataset.hidden_y.values)

            # 初始化 PU 模型
            pu = PU(bagging_pu, PUConfig)

            # 训练 PU 模型
            pu.fit(*dataset.get_train_X_y())

            # 获取真实标签、预测标签和预测概率
            y_true = dataset.y.values
            y_pred = pu.get_predicted_labels()
            y_prob = pu.get_predict_proba()

            # 评估模型
            accuracy, recall, auc, f1 = evaluate_model(y_true, y_pred, y_prob)

            # 将结果存储到列表中
            results.append({
                'dataset': dataset.baseFileName,
                'hidden_rate': member.name,
                'accuracy': accuracy,
                'recall': recall,
                'auc': auc,
                'f1': f1
            })

    # 将结果转换为 DataFrame
    df_results = pd.DataFrame(results)

    # 设置多级索引（一级索引为 dataset，二级索引为 member）
    df_results.set_index(['dataset', 'hidden_rate'], inplace=True)

    return df_results


def semi_supervised_evaluation(dataset):
    """
    对传入的 dataset 进行半监督学习评估，使用 Label Spreading、Label Propagation 和 Self-training 方法。
    结果存储在一个 DataFrame 中，行标签为 HideRatio 的成员名，列标签为方法和评估指标的组合。

    参数:
    - dataset: 包含 df, hidden_y 和 y 属性的对象。

    返回:
    - results_df: 包含评估结果的 DataFrame。
    """
    # 延迟导入
    from datasets.DataSet import DataSet
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
        label_spread = LabelSpreading()
        label_spread.fit(X, hidden_y)

        # 对未标记样本进行预测
        y_pred_spread = label_spread.predict(X_unlabeled)
        y_prob_spread = label_spread.predict_proba(X_unlabeled)[:, 1]

        # 评估 Label Spreading
        accuracy_spread, recall_spread, auc_spread, f1_spread = evaluate_model(y_true_unlabeled, y_pred_spread,
                                                                               y_prob_spread)

        # 2. Label Propagation
        label_prop = LabelPropagation()
        label_prop.fit(X_small, hidden_y_small)

        # 对未标记样本进行预测
        y_pred_prop = label_prop.predict(X_unlabeled)
        y_prob_prop = label_prop.predict_proba(X_unlabeled)[:, 1]

        # 评估 Label Propagation
        accuracy_prop, recall_prop, auc_prop, f1_prop = evaluate_model(y_true_unlabeled, y_pred_prop, y_prob_prop)

        # 3. Self-training
        # 初始化 GBDT 分类器
        gbdt_classifier = GradientBoostingClassifier()
        self_training_model = SelfTrainingClassifier(base_estimator=gbdt_classifier, threshold=0.5)
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


def aggregate_evaluation_results(datasets):
    """
    遍历 datasets 列表，调用 semi_supervised_evaluation 函数，并将结果构造成一个 DataFrame。

    参数:
    datasets (list): 包含多个 dataset 对象的列表。

    返回:
    pd.DataFrame: 包含所有 dataset 评估结果的 DataFrame，一级索引为 dataset.baseFileName。
    """

    # 用于存储所有结果的列表
    all_results = []

    # 遍历每个 dataset
    for dataset in datasets:
        # 调用 semi_supervised_evaluation 函数，获取结果 DataFrame
        results_df = semi_supervised_evaluation(dataset)

        # 为每个结果 DataFrame 添加一列，表示该结果对应的 dataset.baseFileName
        results_df['dataset'] = dataset.baseFileName

        # 将结果存储到 all_results 列表中
        all_results.append(results_df)

    # 将所有结果 DataFrame 合并为一个总的 DataFrame
    final_df = pd.concat(all_results)

    # 将 'dataset' 列设置为索引
    final_df.set_index('dataset', inplace=True)

    return final_df

# 以下方法对外暴露：

def pu_tran_script_run():
    datasets = get_all_dataset()
    result = evaluate_pu_train_datasets(datasets)
    result.to_csv(os.path.join(DATASETS_PATH, 'pu_tran.csv'))

def sklearn_semi_script_run():
    datasets = get_all_dataset()
    result = aggregate_evaluation_results(datasets)
    result.to_csv(os.path.join(DATASETS_PATH, 'SklearnSemiMethodBank.csv'))