import pandas as pd
import numpy as np
from fate.arch.context import create_context

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