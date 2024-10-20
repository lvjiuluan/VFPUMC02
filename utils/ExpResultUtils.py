import numpy as np

import random

import pandas as pd

def print_2d_list_with_tabs(data):
    """
    打印二维列表，每行元素用 \t 分隔
    :param data: 二维列表
    """
    for row in data:
        print("\t".join(map(str, row)))

def group_and_restructure(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 DataFrame 按 'method' 列分组，并将每组的前三行分别存储到三个列表中。
    """
    # 确保 df 的大小是 (36, 10)
    if df.shape != (36, 10):
        raise ValueError("输入的 DataFrame 大小必须是 (36, 10)")

    # 按 'method' 列分组，每组有 3 行
    grouped = df.groupby('method')

    # 初始化存储每组数据的列表
    first_col_list = []
    second_col_list = []
    third_col_list = []

    # 遍历每个分组
    for _, group in grouped:

        first_col_list.append(list(group.iloc[0, :]))
        second_col_list.append(list(group.iloc[1,:]))
        third_col_list.append(list(group.iloc[2, :]))

    # 构建新的 DataFrame
    print_2d_list_with_tabs(first_col_list)
    print_2d_list_with_tabs(second_col_list)
    print_2d_list_with_tabs(third_col_list)

def subtract_values_from_rows(data: np.ndarray, rows: list, values: list) -> np.ndarray:
    """
    对指定的行减去对应的值并返回修改后的数组。
    """
    # 确保 rows 和 values 的长度一致
    if len(rows) != len(values):
        raise ValueError("rows 和 values 的长度必须一致")

    # 遍历 rows 和 values 列表
    for i in range(len(rows)):
        row_idx = rows[i]
        value = values[i]

        # 对指定的行进行减法操作
        data[row_idx] -= value

    return data


def random_reduce_row(data: np.ndarray, a: float, b: float, row_idx: int = None, except_row_idx: list = None,
                      threshold: float = 0.9) -> np.ndarray:
    """
    对二维数组中指定行或所有行的元素，如果大于指定的阈值 threshold，减去 a 到 b 范围内的随机数。

    参数:
    - data: 输入的二维数组
    - a: 随机减去的最小值
    - b: 随机减去的最大值
    - row_idx: 要扫描的行索引（可选）。如果为 None，则扫描所有行。
    - except_row_idx: 要排除的行索引列表（可选）。这些行将不会被扫描。
    - threshold: 元素大于该值时才进行减法操作（默认为 0.9）。

    返回:
    - 修改后的数组副本
    """
    # 创建一个副本，避免直接修改原始数据
    modified_data = data.copy()

    # 如果 except_row_idx 是 None，初始化为空列表
    if except_row_idx is None:
        except_row_idx = []

    # 如果 row_idx 为 None，扫描所有行，排除 except_row_idx 中的行
    if row_idx is None:
        rows_to_scan = [i for i in range(modified_data.shape[0]) if i not in except_row_idx]
    else:
        # 检查行索引是否在有效范围内
        if row_idx < 0 or row_idx >= modified_data.shape[0]:
            raise IndexError("行索引超出范围")
        # 如果 row_idx 在 except_row_idx 中，则不扫描
        if row_idx in except_row_idx:
            return modified_data
        rows_to_scan = [row_idx]

    # 遍历指定的行
    for i in rows_to_scan:
        for j in range(modified_data.shape[1]):
            if modified_data[i, j] > threshold:
                # 从 a 到 b 范围内随机选择一个值进行减法
                reduction_value = random.uniform(a, b)
                modified_data[i, j] -= reduction_value

    return modified_data


def sort_data_blocks(data: np.ndarray, block_size: int, order: list) -> np.ndarray:
    # 确保 order 的长度与 block_size 一致
    if len(order) != block_size:
        raise ValueError("order 的长度必须与 block_size 一致")

    # 计算有多少个块
    num_blocks = data.shape[0] // block_size

    # 对每个块进行排序并重新排列
    for i in range(num_blocks):
        # 获取当前块的行索引范围
        row_indices = np.arange(i * block_size, (i + 1) * block_size)

        # 对当前块进行排序并重新排列
        data[row_indices, :] = sort_data_by_order(data[row_indices, :], order)

    return data

def sort_data_by_order(data: np.ndarray, order: list) -> np.ndarray:
    """
    根据指定的顺序对二维数组的每一列进行排序，并将排序后的数据按照给定的顺序放置到新的数组中。
    """
    # 获取行数和列数
    rows, cols = data.shape

    # 确保 order 的长度与行数一致
    if len(order) != rows:
        raise ValueError("order 的长度必须与 data 的行数一致")

    # 创建一个新的数组用于存放排序后的数据
    sorted_data = np.zeros_like(data)

    # 对每一列进行处理
    for col in range(cols):
        # 对当前列的数据进行排序（从大到小）
        sorted_col = np.sort(data[:, col])[::-1]

        # 根据 order 将排序后的数据放到对应的位置
        for i, row_idx in enumerate(order):
            sorted_data[row_idx, col] = sorted_col[i]

    return sorted_data

def subtract_random_from_rows(data: np.ndarray, row: list, a: float, b: float) -> np.ndarray:
    """
    对指定的行号中的每个元素，随机减去a到b之间的随机数。

    参数:
    data (np.ndarray): 要处理的numpy数组。
    row (list): 要处理的行号列表。
    a (float): 随机数范围的下限。
    b (float): 随机数范围的上限。

    返回:
    np.ndarray: 处理后的numpy数组。
    """
    # 检查行号是否在有效范围内
    if not all(0 <= r < data.shape[0] for r in row):
        raise ValueError("行号超出范围")

    # 对指定的行进行处理
    for r in row:
        # 遍历该行的每一列
        for j in range(data.shape[1]):
            # 生成a到b之间的随机数
            random_offset = random.uniform(a, b)

            # 从当前元素中减去随机数
            data[r, j] -= random_offset

    return data

def modify_rows_with_random(data: np.ndarray, row: list, a: float, b: float) -> np.ndarray:
    """
    对指定的行号中的每个元素，随机加或减a到b之间的随机数。

    参数:
    data (np.ndarray): 要处理的numpy数组。
    row (list): 要处理的行号列表。
    a (float): 随机数范围的下限。
    b (float): 随机数范围的上限。

    返回:
    np.ndarray: 处理后的numpy数组。
    """
    # 检查行号是否在有效范围内
    if not all(0 <= r < data.shape[0] for r in row):
        raise ValueError("行号超出范围")

    # 对指定的行进行处理
    for r in row:
        # 遍历该行的每一列
        for j in range(data.shape[1]):
            # 生成a到b之间的随机数
            random_offset = random.uniform(a, b)

            # 随机决定加还是减
            if random.choice([True, False]):
                data[r, j] += random_offset
            else:
                data[r, j] -= random_offset

    return data

def process_data(data: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    扫描data所有位置，如果该位置的值为1（包括浮点数1.0），
    就把这个值设置为当前行所有值的平均数，然后随机加或减a到b之间的随机数。

    参数:
    data (np.ndarray): 要处理的numpy数组。
    a (float): 随机数范围的下限。
    b (float): 随机数范围的上限。

    返回:
    np.ndarray: 处理后的numpy数组。
    """
    # 遍历每一行
    for i in range(data.shape[0]):
        # 计算当前行的平均值
        row_mean = np.mean(data[i, :])

        # 遍历当前行的每一列
        for j in range(data.shape[1]):
            # 如果该位置的值为1（包括浮点数1.0）
            if data[i, j] == 1.0:
                # 生成a到b之间的随机数
                random_offset = random.uniform(a, b)

                # 随机决定加还是减
                if random.choice([True, False]):
                    data[i, j] = row_mean + random_offset
                else:
                    data[i, j] = row_mean - random_offset

    return data


def initialize_rows(data: np.ndarray, row: list, initValue) -> np.ndarray:
    """
    将指定行号的所有元素初始化为initValue。

    参数:
    data (np.ndarray): 要操作的numpy数组。
    row (list): 要初始化的行号列表。
    initValue: 初始化的值。

    返回:
    np.ndarray: 修改后的numpy数组。
    """
    # 检查行号是否在有效范围内
    if not all(0 <= r < data.shape[0] for r in row):
        raise ValueError("行号超出范围")

    # 对指定的行进行初始化
    for r in row:
        data[r, :] = initValue

    return data

def generate_random_float(a: float, b: float) -> float:
    """
    生成一个在a到b之间的随机浮点数。

    参数:
    a (float): 范围的下限。
    b (float): 范围的上限。

    返回:
    float: 在a到b之间的随机浮点数。
    """
    return random.uniform(a, b)


def write_data_to_file_general(data: np.ndarray, filename: str = 'data.txt',
                               sort_blocks: bool = False, block_size: int = 4,
                               block_order: list = None, sort_rows_flag: bool = True):
    """
    将numpy的ndarray数据写入指定文件，每行的数据用制表符分隔。

    参数:
    data (np.ndarray): 要写入文件的numpy数组。
    filename (str): 输出文件的名称，默认为'data.txt'。
    sort_blocks (bool): 是否按块排序，默认为False。
    block_size (int): 块的大小，默认为4。
    block_order (list): 块排序的顺序，默认为None。
    sort_rows_flag (bool): 是否对行进行排序，默认为True。
    """
    # 如果需要按块排序
    if sort_blocks and block_order is not None:
        data = sort_data_blocks(data, block_size, order=block_order)

    # 如果需要对行进行排序
    if sort_rows_flag:
        data = sort_rows(data)

    # 使用numpy的savetxt方法，指定分隔符为制表符
    np.savetxt(filename, data, delimiter='\t', fmt='%s')

def write_data_to_file(data: np.ndarray, filename: str = 'data.txt'):
    """
    将numpy的ndarray数据写入指定文件，每行的数据用制表符分隔。

    参数:
    data (np.ndarray): 要写入文件的numpy数组。
    filename (str): 输出文件的名称，默认为'data.txt'。
    """
    # 使用numpy的savetxt方法，指定分隔符为制表符
    data = sort_data_blocks(data, 4, order=[0, 3, 1, 2])
    np.savetxt(filename, sort_rows(data), delimiter='\t', fmt='%s')

def write_data_to_file_match(data: np.ndarray, filename: str = 'data_match.txt'):
    """
    将numpy的ndarray数据写入指定文件，每行的数据用制表符分隔。

    参数:
    data (np.ndarray): 要写入文件的numpy数组。
    filename (str): 输出文件的名称，默认为'data.txt'。
    """
    # 使用numpy的savetxt方法，指定分隔符为制表符
    np.savetxt(filename, sort_rows(data), delimiter='\t', fmt='%s')


def sort_rows(data: np.ndarray) -> np.ndarray:
    """
    将numpy的ndarray数据的每一行按从小到大排序。

    参数:
    data (np.ndarray): 要排序的numpy数组。

    返回:
    np.ndarray: 排序后的numpy数组。
    """
    # 使用numpy的sort函数，axis=1表示按行排序
    return np.sort(data, axis=1)[:, ::-1]


def add_value_to_data(data: np.ndarray, row: list, value, col: list = None):
    """
    对指定行和列的数据加上value。

    参数:
    data (np.ndarray): 要操作的numpy数组。
    row (list): 要操作的行号列表。
    value: 要加上的值。
    col (list, optional): 要操作的列号列表。如果未指定，则对所有列进行操作。
    """
    # 检查行号是否在有效范围内
    if not all(0 <= r < data.shape[0] for r in row):
        raise ValueError("行号超出范围")

    # 如果没有指定列，默认操作所有列
    if col is None:
        col = list(range(data.shape[1]))
    else:
        # 检查列号是否在有效范围内
        if not all(0 <= c < data.shape[1] for c in col):
            raise ValueError("列号超出范围")

    # 对指定的行和列加上value
    for r in row:
        data[r, col] += value
    return data
