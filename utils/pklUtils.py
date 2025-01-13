import pickle
import os


def save_to_pkl(obj, pkl_path):
    """
    将对象保存到指定的 Pickle 文件。

    参数：
    - obj: 需要保存的任意 Python 对象。
    - pkl_path (str): 目标 Pickle 文件的路径。

    示例：
    >>> data = {'key': 'value'}
    >>> save_to_pkl(data, 'data.pkl')
    """
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

        with open(pkl_path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"对象已成功保存到 {pkl_path}")
    except Exception as e:
        print(f"保存对象到 {pkl_path} 时发生错误: {e}")


def load_from_pkl(pkl_path):
    """
    从指定的 Pickle 文件加载并返回对象。

    参数：
    - pkl_path (str): 源 Pickle 文件的路径。

    返回：
    - 加载的 Python 对象。

    示例：
    >>> data = load_from_pkl('data.pkl')
    >>> print(data)
    {'key': 'value'}
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"指定的文件 {pkl_path} 不存在。")

    try:
        with open(pkl_path, 'rb') as f:
            obj = pickle.load(f)
        print(f"对象已成功从 {pkl_path} 加载")
        return obj
    except Exception as e:
        print(f"从 {pkl_path} 加载对象时发生错误: {e}")
        raise

def add_to_dict_pkl(dictObj, pkl_path):
    """
    向指定的 Pickle 文件中的字典添加或更新多个键值对。

    参数：
    - dictObj (dict): 要添加或更新的键值对字典。
    - pkl_path (str): 目标 Pickle 文件的路径。

    示例：
    >>> add_to_dict_pkl({'new_key': 'new_value', 'another_key': 'another_value'}, 'data.pkl')
    """
    try:
        # 确保目标目录存在, 这一步不会创建文件！ 而是创建文件目录
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

        # 初始化字典
        existing_data = {}

        # 如果文件存在，读取现有数据
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                existing_data = pickle.load(f)
                if not isinstance(existing_data, dict):
                    raise ValueError("Pickle 文件中的数据不是字典类型。")

        # 更新字典数据
        existing_data.update(dictObj)

        # 将更新后的字典写回文件
        with open(pkl_path, 'wb') as f:
            pickle.dump(existing_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"字典已成功更新到 {pkl_path}")
    except Exception as e:
        print(f"更新字典到 {pkl_path} 时发生错误: {e}")