import os
import subprocess
import sys

import pandas as pd
import numpy as np
from fate.arch.context import create_context
import yaml
import inspect
from fate.arch.dataframe import PandasReader
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from sklearn.model_selection import train_test_split

from consts.Constants import *
from enums.SbtObjective import SbtObjective
from utils.Logger import Logger
from utils.pklUtils import *
import ast
import json

logger = Logger.get_logger()

def fate_construct_df(XA, XB, y=None):
    """
    构造两个 DataFrame: A_df 和 B_df。

    Args:
        XA (numpy.ndarray): 用于构造 A_df 的数据。
        XB (numpy.ndarray): 用于构造 B_df 的数据。
        y (numpy.ndarray, optional): 用于构造 B_df 的 'y' 列的数据。如果为 None，则不包含 'y' 列。

    Returns:
        tuple: 包含 A_df 和 B_df 的元组。
    """
    len_XA = len(XA)
    len_XB = len(XB)
    p = XA.shape[1] if len_XA > 0 and XA.ndim > 1 else 0
    q = XB.shape[1] if len_XB > 0 and XB.ndim > 1 else 0

    # 构造 A_df
    sample_ids_a = list(range(len_XA))
    ids_a = list(range(len_XA))
    columns_a = ['sample_id', 'id'] + [f'x{i}' for i in range(p)]
    data_a = {'sample_id': sample_ids_a, 'id': ids_a}
    for i in range(p):
        data_a[f'x{i}'] = XA[:, i]
    A_df = pd.DataFrame(data_a)

    # 构造 B_df
    sample_ids_b = list(range(len_XB))
    ids_b = list(range(len_XB))
    columns_b = ['sample_id', 'id'] + [f'x{i}' for i in range(q)]
    data_b = {'sample_id': sample_ids_b, 'id': ids_b}
    if y is not None:
        columns_b.append('y')
        data_b['y'] = y
    for i in range(q):
        data_b[f'x{i}'] = XB[:, i]
    B_df = pd.DataFrame(data_b)

    return A_df, B_df


def create_ctx(party, session_id='test_fate'):
    parties = [("guest", "9999"), ("host", "10000")]
    if party == "guest":
        local_party = ("guest", "9999")
    else:
        local_party = ("host", "10000")
    context = create_context(local_party, parties=parties, federation_session_id=session_id)
    return context


def create_host_guest_ctx(session_id='test_fate'):
    """
    直接创建 guest 和 host 两个 party 的 context。

    Args:
        session_id (str, optional): Federation session ID. Defaults to 'test_fate'.

    Returns:
        tuple: 包含 guest 和 host 两个 context 的元组，顺序为 (guest_context, host_context)。
    """
    guest_ctx = create_ctx("guest", session_id=session_id)
    host_ctx = create_ctx("host", session_id=session_id)
    return host_ctx, guest_ctx


def load_config(filepath):
    """
    读取 sbtConfig.yaml 文件并将其内容转换为字典。

    Args:
        filepath (str): sbtConfig.yaml 文件的路径。默认为 "sbtConfig.yaml"。

    Returns:
        dict: 从 YAML 文件加载的配置字典。
              如果文件不存在或加载失败，则返回一个空字典。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return {}
    except yaml.YAMLError as e:
        print(f"错误：解析 YAML 文件时出错： {e}")
        return {}


def save_config(config: dict, filepath: str = "sbtConfig.yaml"):
    """
    将字典类型的配置信息保存到 YAML 文件。

    Args:
        config (dict): 需要保存的配置字典。
        filepath (str): 保存的目标文件路径，默认为 "sbtConfig.yaml"。
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # safe_dump 会将数据写入 YAML 格式文件
            # sort_keys=False 保证字典的键顺序不被自动排序（可根据需要开启或关闭）
            # allow_unicode=True 可以在文件中直接写入非 ASCII 字符
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        print(f"配置已成功保存到 {filepath}")
    except Exception as e:
        print(f"保存到 YAML 文件时出错：{e}")


def save_host_guest_dataframes(A_df: pd.DataFrame, B_df: pd.DataFrame, A_host_path: str, B_guest_path: str):
    """
    将两个 DataFrame（分别代表 host 和 guest 数据）保存到指定的 CSV 文件路径。

    Args:
        A_df (pd.DataFrame): 需要保存的 host 数据。
        B_df (pd.DataFrame): 需要保存的 guest 数据。
        A_host_path (str): host DataFrame 的目标 CSV 文件路径。
        B_guest_path (str): guest DataFrame 的目标 CSV 文件路径。

    示例：
        >>> A_host_path = os.path.join(DATASETS_PATH, 'A_host.csv')
        >>> B_guest_path = os.path.join(DATASETS_PATH, 'B_guest.csv')
        >>> save_host_guest_dataframes(A_df, B_df, A_host_path, B_guest_path)
    """
    try:
        # 保存 host DataFrame 到 A_host_path
        save_df_to_csv(A_df, A_host_path)

        # 保存 guest DataFrame 到 B_guest_path
        save_df_to_csv(B_df, B_guest_path)

        print(f"两个 DataFrame 已成功保存到：\n  {A_host_path}\n  {B_guest_path}")
    except Exception as e:
        print(f"保存 DataFrame 时发生错误：{e}")


def save_df_to_csv(df: pd.DataFrame, file_path: str):
    """
    将单个 DataFrame 保存到指定的 CSV 文件路径。

    Args:
        df (pd.DataFrame): 要保存的 DataFrame。
        file_path (str): 目标 CSV 文件的路径。
    """
    try:
        # 将 DataFrame 保存为 CSV 文件
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"DataFrame 已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存 DataFrame 到 {file_path} 时发生错误：{e}")
        raise


def load_host_guest_data(A_host_path: str, B_guest_path: str, skip_columns: list):
    """
    从两个 CSV 文件中加载数据，跳过指定的列，并返回特征和标签的 NumPy 数组。

    Args:
        A_host_path (str): `A_host.csv` 文件的路径。
        B_guest_path (str): `B_guest.csv` 文件的路径。
        skip_columns (List[str]): 要跳过的列名列表。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - XA: 从 `A_host.csv` 提取的特征数据。
            - XB: 从 `B_guest.csv` 提取的特征数据。
            - y: 从 `B_guest.csv` 提取的标签数据。

    Raises:
        FileNotFoundError: 如果任一文件不存在。
        ValueError: 如果 `'y'` 列在 `B_guest.csv` 中不存在。
    """
    # 检查文件是否存在
    if not os.path.isfile(A_host_path):
        raise FileNotFoundError(f"文件不存在: {A_host_path}")
    if not os.path.isfile(B_guest_path):
        raise FileNotFoundError(f"文件不存在: {B_guest_path}")

    # 读取 CSV 文件到 DataFrame
    A_df = pd.read_csv(A_host_path)
    B_df = pd.read_csv(B_guest_path)

    # 打印读取成功的消息（可选）
    print(f"成功读取 {A_host_path} 和 {B_guest_path}")

    # 检查 'y' 列是否存在于 B_df 中
    if 'y' not in B_df.columns:
        raise ValueError("在 `B_guest.csv` 中未找到标签列 'y'")

    # 删除指定的列，如果这些列存在
    A_columns_before = set(A_df.columns)
    B_columns_before = set(B_df.columns)

    A_df = A_df.drop(columns=[col for col in skip_columns if col in A_df.columns], errors='ignore')
    B_df = B_df.drop(columns=[col for col in skip_columns if col in B_df.columns], errors='ignore')

    # 打印跳过的列信息（可选）
    skipped_A = set(skip_columns).intersection(A_columns_before)
    skipped_B = set(skip_columns).intersection(B_columns_before)
    if skipped_A:
        print(f"A_host.csv 中跳过的列: {skipped_A}")
    if skipped_B:
        print(f"B_guest.csv 中跳过的列: {skipped_B}")

    # 提取标签 y
    y = B_df['y'].values

    # 删除 'y' 列以获取特征
    B_features_df = B_df.drop(columns=['y'])

    # 转换 DataFrame 为 NumPy 数组
    XA = A_df.values
    XB = B_features_df.values

    return XA, XB, y


def filter_params_for_class(cls, config):
    """
    过滤掉与类的构造函数无关的参数。

    :param cls: 类名
    :param config: 包含参数的字典
    :return: 过滤后的字典，只包含构造函数所需的参数
    """
    # 获取构造函数的参数名
    constructor_params = inspect.signature(cls.__init__).parameters

    # 过滤掉 config 中与构造函数无关的键值对
    filtered_config = {k: v for k, v in config.items() if k in constructor_params}

    return filtered_config


def df_to_data(ctx, df, has_label=True):
    if has_label:
        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", label_name="y", dtype="float32")
    else:
        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", dtype="float32")

    fate_df = reader.to_frame(ctx, df)
    return fate_df


def execute_sbt_command(config):
    """
    执行 SBT 脚本命令，检查执行结果，并加载 guest 和 host 的结果文件。

    参数：
    - config (dict): 配置字典，包含 `log_level` 等信息。
    - sbt_script_path (str): SBT 脚本的路径。
    - sbt_pkl_guest_path (str): guest 结果的 Pickle 文件路径。
    - sbt_pkl_host_path (str): host 结果的 Pickle 文件路径。

    返回：
    - dict: 包含 guest 和 host 结果的字典。
      格式为：{'guest': guest_result, 'host': host_result}
    """
    try:
        # 构建命令
        command = [
            sys.executable,
            SBT_SCRIPT_PATH,
            '--parties',
            'guest:9999',
            'host:10000',
            '--log_level',
            config['log_level']
        ]

        # 使用 subprocess.run 执行命令
        result = subprocess.run(command, capture_output=True, text=True)

        # 打印输出结果
        print("Standard Output:", result.stdout)
        print("Standard Error:", result.stderr)

        # 检查命令是否成功运行
        if result.returncode != 0:
            print("Command failed with return code:", result.returncode)
            raise RuntimeError(f"Command execution failed with return code {result.returncode}")

        print("Command executed successfully.")

        # 加载 guest 和 host 的结果
        guest_result = load_from_pkl(SBT_PKL_GUEST_PATH)
        host_result = load_from_pkl(SBT_PKL_HOST_PATH)

        # 返回结果
        return {
            'guest': guest_result,
            'host': host_result
        }
    except Exception as e:
        print(f"An error occurred during command execution: {e}")
        raise


def parse_probability_details(detail):
    """
    将包含字符串表示的字典的数组转换为一个二维numpy数组。
    
    参数:
    detail -- numpy数组，每个元素是一个表示类别概率的字典的字符串。
    C -- int，表示类别的数量。
    
    返回:
    numpy数组，形状为(n, C)，其中n是detail的行数。
    """
    probabilities = []

    # 从第一个元素推断类别数量
    first_item = ast.literal_eval(json.loads(detail[0]))
    C = len(first_item)  # 类别数量为字典的键的数量

    # 遍历detail数组中的每个元素
    for item in detail:
        # 将字符串表示的字典转换为真正的字典
        prob_dict = ast.literal_eval(json.loads(item))
        # 提取概率值并按类别顺序添加到列表
        prob_values = [prob_dict[str(i)] for i in range(C)]
        probabilities.append(prob_values)

    # 将列表转换为numpy数组
    prob_array = np.array(probabilities)

    return prob_array


def remove_file_extension(file_name, new_suffix='.py'):
    """
    去掉文件名后缀，并根据需要添加新的后缀。

    :param file_name: 文件名字符串
    :param new_suffix: 新的后缀（默认为 '.py'），如果为 None，则不添加新的后缀
    :return: 去掉旧后缀并添加新后缀的文件名
    """
    # 去掉旧的后缀
    if '.' in file_name:
        base_name = file_name.rsplit('.', 1)[0]
    else:
        base_name = file_name

    # 如果 new_suffix 不为 None，则拼接新的后缀
    if new_suffix is not None:
        if not new_suffix.startswith('.'):
            new_suffix = '.' + new_suffix  # 确保后缀以 "." 开头
        return base_name + new_suffix

    # 如果 new_suffix 为 None，直接返回去掉后缀的文件名
    return base_name


def convert_ipynb_to_py(ipynb_file_name, py_file_name=None):
    """
    将 Jupyter Notebook 文件 (.ipynb) 转换为 Python 脚本 (.py) 文件。

    参数:
        ipynb_file_name (str): 输入的 Jupyter Notebook 文件路径，必须是 .ipynb 文件。
        py_file_name (str, 可选): 输出的 Python 脚本文件名。如果未提供该参数，
                                  则会根据 `ipynb_file_name` 自动生成一个相应的 .py 文件名。

    返回:
        无返回值。此方法会将转换后的 Python 脚本保存在指定的路径中。

    注意事项:
        1. 该方法假定输入的 Jupyter Notebook 文件是有效的 JSON 格式。
        2. 输出的 Python 脚本中仅包含代码单元（code cells）的内容，其他单元（如 markdown 单元）将被忽略。
        3. 如果未提供 `py_file_name`，生成的 Python 文件名会基于 `ipynb_file_name` 的名称，
           并保存在 `SCRIPTS_PATH` 目录中。
        4. 需要确保 `SCRIPTS_PATH` 是有效的路径，且程序具有写入权限。

    示例:
        >>> convert_ipynb_to_py("example_notebook.ipynb", "example_script.py")
        # 将 example_notebook.ipynb 转换为 example_script.py

        >>> convert_ipynb_to_py("example_notebook.ipynb")
        # 自动生成 example_notebook.py 文件，并保存在 SCRIPTS_PATH 目录下
    """
    with open(ipynb_file_name, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    if py_file_name is None:
        py_file_path = os.path.join(SCRIPTS_PATH, remove_file_extension(ipynb_file_name, new_suffix='.py'))
    else:
        py_file_path = os.path.join(SCRIPTS_PATH, py_file_name)
    with open(py_file_path, 'w', encoding='utf-8') as f:
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                f.write(''.join(cell['source']) + '\n\n')


def determine_task_type(y: np.ndarray):
    """
    根据给定的目标向量 y 判断当前任务类型（二分类、多分类或回归），
    并返回对应的目标函数名称 (SbtObjective.value) 和类别数（回归任务为 None）。

    参数:
    --------
    y : np.ndarray
        目标向量，通常是一维数组。若属于分类任务，则应包含离散的整型标签，
        若为回归任务，则可包含连续值或难以归为有限类别的值。

    返回:
    --------
    objective_value : str
        对应的目标函数名称，来自 SbtObjective 枚举的 value。
    num_class : int or None
        对于二分类和多分类任务，返回类别数量；对于回归任务，返回 None。

    示例:
    --------
    >>> y_class_binary = np.array([0, 1, 0, 1])
    >>> determine_task_type(y_class_binary)
    ('binary:bce', 2)

    >>> y_class_multi = np.array([0, 1, 2, 1, 3])
    >>> determine_task_type(y_class_multi)
    ('multi:ce', 4)

    >>> y_reg = np.array([3.14, 2.71, 1.41, 2.0])
    >>> determine_task_type(y_reg)
    ('regression:l2', None)
    """
    # 去重后得到的唯一值
    unique_values = np.unique(y)
    num_unique = len(unique_values)

    print(f"输入标签 y 的唯一值个数: {num_unique}, 唯一值: {unique_values}")

    # 判断是否为整数类型（用于区分离散和连续）
    # 你也可以根据具体业务需求进行更复杂的判断
    if issubclass(y.dtype.type, np.integer):
        if num_unique == 2:
            print("检测到二分类任务 (Binary Classification)")
            objective_value = SbtObjective.BINARY_BCE.value
            num_class = 2
        elif num_unique > 2:
            print("检测到多分类任务 (Multi-class Classification)")
            objective_value = SbtObjective.MULTI_CE.value
            num_class = num_unique
        else:
            # num_unique == 1 或者其他极端情况
            logger.warning("仅检测到 1 个唯一值，或未能判定为常见分类任务，默认回归处理")
            objective_value = SbtObjective.REGRESSION_L2.value
            num_class = None
    else:
        # 若不是整数类型，则默认回归处理
        print("检测到回归任务 (Regression)")
        objective_value = SbtObjective.REGRESSION_L2.value
        num_class = None

    return objective_value, num_class



def get_data_by_objective(objective: SbtObjective):
    """
    根据给定的枚举类型 (objective)，返回对应的 X, y 数据。
    - BINARY_BCE: 使用乳腺癌数据集 (二分类)
    - MULTI_CE: 使用鸢尾花数据集 (多分类)
    - REGRESSION_L2: 使用糖尿病数据集 (回归)
    """
    if objective == SbtObjective.BINARY_BCE:
        data = load_breast_cancer()
    elif objective == SbtObjective.MULTI_CE:
        data = load_iris()
    elif objective == SbtObjective.REGRESSION_L2:
        data = load_diabetes()
    else:
        raise ValueError(f"Unsupported objective: {objective}")

    X, y = data.data, data.target

    # 计算需要拆分的索引（列的中间位置）
    n_col = X.shape[1]
    split_idx = n_col // 2  # 整数除法

    # 划分 XA, XB
    XA = X[:, :split_idx]
    XB = X[:, split_idx:]

    # 划分训练集和测试集
    XA_train, XA_test, XB_train, XB_test, y_train, y_test = train_test_split(XA, XB, y, test_size=0.2, random_state=42)
    return XA_train, XA_test, XB_train, XB_test, y_train, y_test


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report, mean_squared_error,
    mean_absolute_error, r2_score, explained_variance_score
)

def evaluate_model(y_true, y_pred, y_proba=None, task='classification'):
    """
    根据任务类型（分类或回归）打印模型评估指标。

    参数：
    ----------
    y_true : array-like
        真实标签（分类任务）或真实值（回归任务）。
    y_pred : array-like
        分类任务：预测的离散标签；
        回归任务：预测的数值结果。
    y_proba : array-like, 可选
        在分类任务下，预测的概率值（通常是正类的概率或多类概率）。
        当需要计算 ROC AUC 等基于概率的指标时，需要此参数。
        回归任务无需此参数。
    task : str, 可选
        指定任务类型，可选 'classification' 或 'regression'，默认 'classification'。
    """

    if task == 'classification':
        print("\n========== Classification Metrics ==========")

        # --------------------
        # 1. 基础分类指标
        # --------------------
        acc = accuracy_score(y_true, y_pred)
        pre_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"Accuracy         : {acc:.4f}")
        print(f"Precision (macro): {pre_macro:.4f}")
        print(f"Recall (macro)   : {rec_macro:.4f}")
        print(f"F1 (macro)       : {f1_macro:.4f}")

        # --------------------
        # 2. 混淆矩阵
        # --------------------
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # --------------------
        # 3. ROC AUC
        # --------------------
        # 如果需要并且提供了预测概率，可以计算二分类或多分类的 ROC AUC
        if y_proba is not None:
            try:
                # 对二分类情况，可直接计算
                # 若是多分类，可指定 multi_class='ovr' 或 'ovo' 并传入相应格式的 y_proba
                auc_score = roc_auc_score(y_true, y_proba)
                print(f"\nROC AUC          : {auc_score:.4f}")
            except ValueError:
                print("\nROC AUC          : 无法计算（可能是多分类且未指定multi_class）")

        # --------------------
        # 4. Classification Report
        # --------------------
        print("\nClassification Report:")
        cls_report = classification_report(y_true, y_pred, zero_division=0)
        print(cls_report)

    elif task == 'regression':
        print("\n========== Regression Metrics ==========")

        # --------------------
        # 1. 回归指标
        # --------------------
        mse_val = mean_squared_error(y_true, y_pred)
        rmse_val = np.sqrt(mse_val)
        mae_val = mean_absolute_error(y_true, y_pred)
        r2_val = r2_score(y_true, y_pred)
        evs_val = explained_variance_score(y_true, y_pred)

        print(f"MSE               : {mse_val:.4f}")
        print(f"RMSE              : {rmse_val:.4f}")
        print(f"MAE               : {mae_val:.4f}")
        print(f"R^2 (Coefficient) : {r2_val:.4f}")
        print(f"Explained Variance: {evs_val:.4f}")

    else:
        raise ValueError("task 参数必须是 'classification' 或 'regression'。")