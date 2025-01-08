import numpy as np
import logging

# 启用日志记录
logging.basicConfig(level=logging.WARNING)

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


def test_get_top_k_percent_idx():
    # 测试示例 1：手动构造的简单数组
    scores1 = np.array([1, 3, 2, 5, 4])
    k1 = 0.4  # 取前 40%
    # 预期：长度为 5 * 0.4 = 2，向下取整为 2；如果取最大的前 40%，就是取 top2 分数；如果取最小的，则取 min2。

    idx1_desc = get_top_k_percent_idx(scores1, k1, pick_lowest=False)
    idx1_asc = get_top_k_percent_idx(scores1, k1, pick_lowest=True)
    print("示例1 (scores=[1,3,2,5,4], k=0.4, pick_lowest=False)  ==> 索引:", idx1_desc, " 对应分数:", scores1[idx1_desc])
    print("示例1 (scores=[1,3,2,5,4], k=0.4, pick_lowest=True)   ==> 索引:", idx1_asc, " 对应分数:", scores1[idx1_asc])
    print("-" * 50)

    # 测试示例 2：k=0 时，仍然至少返回 1 个元素
    scores2 = np.array([10, 20, 30, 40, 50])
    k2 = 0.0
    # 预期：由于 k=0 => int(n*k)=0，但函数里会强制至少取 1 个。
    idx2_desc = get_top_k_percent_idx(scores2, k2, pick_lowest=False)
    idx2_asc = get_top_k_percent_idx(scores2, k2, pick_lowest=True)
    print("示例2 (scores=[10,20,30,40,50], k=0.0, pick_lowest=False) ==> 索引:", idx2_desc, " 对应分数:", scores2[idx2_desc])
    print("示例2 (scores=[10,20,30,40,50], k=0.0, pick_lowest=True)  ==> 索引:", idx2_asc, " 对应分数:", scores2[idx2_asc])
    print("-" * 50)

    # 测试示例 3：k=1 时，应该返回全部元素
    scores3 = np.array([10, 20, 30, 40, 50])
    k3 = 1.0
    idx3_desc = get_top_k_percent_idx(scores3, k3, pick_lowest=False)
    idx3_asc = get_top_k_percent_idx(scores3, k3, pick_lowest=True)
    print("示例3 (scores=[10,20,30,40,50], k=1.0, pick_lowest=False) ==> 索引:", idx3_desc, " 对应分数:", scores3[idx3_desc])
    print("示例3 (scores=[10,20,30,40,50], k=1.0, pick_lowest=True)  ==> 索引:", idx3_asc, " 对应分数:", scores3[idx3_asc])
    print("-" * 50)

    # 测试示例 4：随机浮点数组测试
    np.random.seed(42)  # 固定随机种子，便于复现
    scores4 = np.random.rand(10) * 100  # 10 个随机分数，范围 [0,100)
    k4 = 0.3  # 取前 30%

    idx4_desc = get_top_k_percent_idx(scores4, k4, pick_lowest=False)
    idx4_asc = get_top_k_percent_idx(scores4, k4, pick_lowest=True)
    print("示例4 (随机分数, k=0.3, pick_lowest=False) ==> 索引:", idx4_desc, " 对应分数:", scores4[idx4_desc])
    print("示例4 (随机分数, k=0.3, pick_lowest=True)  ==> 索引:", idx4_asc, " 对应分数:", scores4[idx4_asc])
    print("-" * 50)

    # 测试示例 5：大数组的性能和正确性（只做简单示例演示）
    scores5 = np.random.rand(1000000)  # 100 万个随机数
    k5 = 0.001  # 取前 0.1%
    # 只要能正确返回索引长度，就说明函数在大数组下的表现是可行的
    idx5_desc = get_top_k_percent_idx(scores5, k5, pick_lowest=False)
    idx5_asc = get_top_k_percent_idx(scores5, k5, pick_lowest=True)
    print("示例5 (大数组, k=0.001, pick_lowest=False) ==> 长度:", len(idx5_desc))
    print("示例5 (大数组, k=0.001, pick_lowest=True)  ==> 长度:", len(idx5_asc))


test_get_top_k_percent_idx()