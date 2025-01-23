from classfiers.VF_BASE import VF_BASE_CLF
from utils.FateUtils import determine_task_type

from vf4lr.client import Client
from vf4lr.server import Server
from vf4lr.train import vfl_lr_train, evaluation_get

from utils.Logger import logger


class VF_LR(VF_BASE_CLF):
    """
    VF_LR 类用于垂直联邦学习场景下的逻辑回归（Logistic Regression）模型。

    参数：
    ----------
    learning_rate : float
        学习速率，默认值为 3。
    epoch_num : int
        训练轮数，默认值为 5。
    batch_size : int
        每批次训练的样本数量，默认值为 128。

    属性：
    ----------
    config : dict
        存储模型配置，包括学习率、训练轮数、批大小、客户端数量、类别数量等。
    _is_fitted : bool
        标志模型是否训练完成。若为 True，则表示模型已经被训练，可直接进行预测。
    X_trains : list
        存储训练数据的列表。X_trains[0] 和 X_trains[1] 分别对应两个参与方的训练数据。
    X_test_s : list
        存储预测数据的列表。X_test_s[0] 和 X_test_s[1] 分别对应两个参与方的预测数据。
    Y_train : array-like
        训练标签。
    y_pred : array-like
        预测结果标签。
    y_proba : array-like
        预测结果的概率分布，多列表示不同类别的概率。
    """

    def __init__(self, learning_rate=0.2, epoch_num=5, batch_size=64):
        """
        初始化 VF_LR 模型。

        参数：
        ----------
        learning_rate : float
            学习速率，默认值为 3。
        epoch_num : int
            训练轮数，默认值为 5。
        batch_size : int
            每批次训练的样本数量，默认值为 128。
        """
        self.config = {
            'learning_rate': learning_rate,
            'epoch_num': epoch_num,
            'batch_size': batch_size,
            'client_num': 2  # 固定为2个客户端
        }

        # 日志记录
        logger.info("VF_LR 模型初始化完成。")
        logger.info("模型配置: %s", self.config)

    def fit(self, XA, XB, y):
        """
        训练垂直联邦逻辑回归模型。

        参数：
        ----------
        XA : array-like
            客户端 A 的训练数据。
        XB : array-like
            客户端 B 的训练数据。
        y : array-like
            训练标签。
        """
        # 获取数据任务类型（分类或回归）以及类别数量
        _, class_num = determine_task_type(y)
        self.config['class_num'] = class_num

        # 存储训练数据和标签
        self.Y_train = y
        self.X_trains = [XA, XB]

        # 打印日志：显示当前训练数据的形状与类别数
        logger.info(f"开始训练垂直联邦逻辑回归模型，客户端A训练数据形状: {XA.shape}, 客户端B训练数据形状: {XB.shape}, 标签大小: {len(y)}")
        logger.info(f"检测到的类别数量: {class_num}")
        logger.info("模型配置："
                    f"学习率={self.config['learning_rate']}, "
                    f"训练轮数={self.config['epoch_num']}, "
                    f"批大小={self.config['batch_size']}, "
                    f"客户端数={self.config['client_num']}")

        # 在此处可进行更多的前置检查或数据处理
        logger.info("完成模型初始化，等待调用 predict 或 predict_proba 进行训练和预测。")

    def predict(self, XA, XB):
        """
        使用训练好的模型进行预测并返回分类结果（标签）。

        参数：
        ----------
        XA : array-like
            客户端 A 的测试数据。
        XB : array-like
            客户端 B 的测试数据。

        返回：
        ----------
        array-like
            预测的标签结果。
        """
        # 设置测试数据
        self.X_test_s = [XA, XB]
        self.config['test_size'] = len(XA)  # 仅需由其中一个的长度表示测试集大小

        # 日志
        logger.info(f"开始进行预测，客户端A测试数据形状: {XA.shape}, 客户端B测试数据形状: {XB.shape}")
        return self._execute_prediction(XA, XB, return_proba=False)

    def predict_proba(self, XA, XB):
        """
        使用训练好的模型进行预测并返回预测概率。
        返回多维数组，每一列表示某一类别的预测概率。

        参数：
        ----------
        XA : array-like
            客户端 A 的测试数据。
        XB : array-like
            客户端 B 的测试数据。

        返回：
        ----------
        array-like
            预测概率，多列表示不同类别的概率。
        """
        # 设置测试数据
        self.X_test_s = [XA, XB]
        self.config['test_size'] = len(XA)

        # 日志
        logger.info(f"开始进行预测概率计算，客户端A测试数据形状: {XA.shape}, 客户端B测试数据形状: {XB.shape}")
        return self._execute_prediction(XA, XB, return_proba=True)

    def _execute_prediction(self, XA, XB, return_proba):
        """
        内部方法，执行预测过程。当模型尚未真正训练时，会触发一次实际的训练流程。

        参数：
        ----------
        XA : array-like
            客户端 A 的数据（用于测试）。
        XB : array-like
            客户端 B 的数据（用于测试）。
        return_proba : bool
            是否返回预测概率。如果为 True，则返回预测的概率分布，否则返回预测的标签。

        返回：
        ----------
        array-like
            如果 return_proba = False，返回预测的标签；
            如果 return_proba = True，返回预测的概率分布。
        """

        # 读取训练数据和配置
        Y_train, config = self.Y_train, self.config
        X_train_s = self.X_trains
        X_test_s = self.X_test_s

        # 初始化 server
        logger.info("初始化 Server 对象。")
        server = Server(Y_train, config)

        # 初始化若干客户端 Client
        logger.info("初始化 Client 对象。")
        clients = []
        for i in range(config['client_num']):
            c = Client(X_train_s[i], X_test_s[i], config)
            c.set_id(i)
            clients.append(c)
            logger.info(f"Client {i} 初始化完成，训练数据形状: {X_train_s[i].shape}, 测试数据形状: {X_test_s[i].shape}")

        # 将客户端挂载到 Server
        server.attach_clients(clients)
        logger.info("所有客户端挂载至 Server 完成，开始进行联邦训练。")

        # 进行联邦训练
        vfl_lr_train(server, clients)
        logger.info("联邦训练完成。")

        # 获取训练结果：预测标签和预测概率
        self.y_proba, self.y_pred = evaluation_get(server, clients)
        logger.info("获取最终预测结果。")

        logger.info("模型训练完成并已缓存预测结果。")

        # 根据需求返回预测标签或预测概率
        if return_proba:
            logger.info(f"返回预测概率，形状: {self.y_proba.shape}")
            return self.y_proba
        else:
            logger.info(f"返回预测标签，形状: {self.y_pred.shape}")
            return self.y_pred
