from typing import List
from vf4lr.server import Server, Client
from vf4lr.util import *
from typing import List

from utils.Logger import logger


def vfl_lr_train(server: Server, clients: List[Client]):
    """
    垂直联邦学习逻辑回归训练流程:
    1. 将训练数据按照 batch_size 进行分批；
    2. 服务端和客户端协同训练, 依次完成以下步骤:
       - 服务端向客户端请求并更新 embedding 信息；
       - 服务端计算损失和梯度, 并将梯度广播给客户端；
       - 服务端更新自身参数(例如偏置)；
       - 客户端更新自身模型参数(例如权重)。
    """
    logger.info(f"开始 VFL LR 训练，共 {server.epoch_num} 轮。")

    for epoch in range(server.epoch_num):
        logger.info(f"开始第 {epoch + 1}/{server.epoch_num} 轮训练...")

        # 计算每轮要训练的批次数量
        batch_num = server.data_num // server.batch_size
        # 将数据划分成若干批次索引列表
        batches = gen_batches(server.data_num, server.batch_size)

        for batch_idx in range(batch_num):
            # 当前批次的数据索引
            batch_indexes = batches[batch_idx]

            # 日志: 当前批次的信息
            logger.debug(f"第 {epoch + 1} 轮, 批次 {batch_idx + 1}/{batch_num}, 批次大小: {len(batch_indexes)}")

            # 第一步：设置各个客户端的批次索引，并请求更新其 embedding 信息
            for c in clients:
                c.set_batch_indexes(batch_indexes)
                server.update_embedding_data(c)  # 向客户端请求 embedding 并进行更新

            # 第二步：服务端基于所有客户端的 embedding 计算当前批次的损失和梯度，并将梯度广播给各客户端
            loss = server.cal_batch_embedding_grads()
            logger.debug(f"当前批次损失值: {loss}")

            # 第三步：服务端更新自身模型参数(如偏置项)
            server.update_bias()

            # 第四步：客户端更新各自的模型参数(如权重)
            for c in clients:
                c.update_weight()

        logger.info(f"第 {epoch + 1} 轮训练完成。")

    logger.info("所有轮次训练完成。")


def evaluation_get(server: Server, clients: List[Client]):
    # 在测试数据集上展示模型性能
    y_proba = []  # 用于存储每个样本的预测概率
    y_pred = []  # 用于存储每个样本的预测结果

    # 从每个客户端更新服务器的嵌入数据，用于测试期间
    for c in clients:
        server.update_embedding_data(c, period_type="test")

    # 聚合所有客户端的嵌入数据
    aggr_embedding_data = np.sum(server.test_embedding_data, axis=0)

    # 遍历测试集中的每个样本
    for idx in range(server.test_size):
        # 使用 softmax 计算预测概率
        pred_prob = softmax(aggr_embedding_data[:, idx] + server.bias)

        # 存储预测概率以供后续使用
        y_proba.append(pred_prob)

        # 确定预测的类别
        pred_class = np.argmax(pred_prob)

        # 存储预测的类别
        y_pred.append(pred_class)

    # 将 y_proba 和 y_pred 转换为 numpy 数组，以保持一致性
    y_proba = np.array(y_proba)
    y_pred = np.array(y_pred)

    return y_proba, y_pred
