import os

import pandas as pd
import numpy as np
from fate.arch.context import create_context
import yaml


def fate_construct_df(XA, XB, y):
    """
    构造两个 DataFrame: A_df 和 B_df。

    Args:
        XA (numpy.ndarray): 用于构造 A_df 的数据。
        XB (numpy.ndarray): 用于构造 B_df 的数据。
        y (numpy.ndarray): 用于构造 B_df 的 'y' 列的数据。

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
    columns_b = ['sample_id', 'id', 'y'] + [f'x{i}' for i in range(q)]
    data_b = {'sample_id': sample_ids_b, 'id': ids_b, 'y': y}
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
