import math
import time
import logging
from utils.SimpleHomomorphicEncryption import SimpleHomomorphicEncryption
import numpy as np
from utils.Logger import Logger

from classfiers.VF_BASE import VF_BASE_CLF

logger = Logger.get_logger()


class VF_LR(VF_BASE_CLF):
    def __init__(self, config):
        self.config = config
        self.weightA = None
        self.weightB = None
        self.loss_history = []

        # 日志记录
        logger.info("VF_LinearRegression 模型初始化完成。")
        logger.info("模型配置: %s", self.config)

    def fit(self, XA, XB, y):

        logger.info("开始构建A、B、C三个客户端...")
        # 初始化客户端对象
        client_a = ClientA(XA, self.config)
        client_b = ClientB(XB, y, self.config)
        client_c = ClientC(XA.shape, XB.shape, self.config)

        # 建立客户端之间的连接
        for client, name in zip([client_a, client_b, client_c], ['A', 'B', 'C']):
            logger.info(f"正在为客户端{name}建立连接...")
            for target_name, target_client in zip(['A', 'B', 'C'], [client_a, client_b, client_c]):
                if client is not target_client:
                    client.connect(target_name, target_client)

        # 查看连接信息
        logger.info("查看各客户端的连接情况:")
        for client, name in zip([client_a, client_b, client_c], ['A', 'B', 'C']):
            logger.info(f"客户端{name}已连接: {list(client.other_client.keys())}")

        # 开始迭代训练
        logger.info("开始训练迭代流程...")
        for iteration in range(self.config['num_iters']):
            # 1. C方创建钥匙对，分发公钥给A和B
            client_c.generate_and_distribute_keys('A', 'B')

            # 2. A方计算[[u_a]]和[[u_a^2]]发送给B方
            client_a.compute_u_a_and_squares('B')

            # 3. B方计算[[u_b]]、[[u_b^2]]、[[L]]、[[d]]、[[masked_dL_b]]并将[[d]]发送给A方，[[L]]和[[masked_dL_b]]发送给C方
            client_b.compute_gradient_and_loss('A', 'C')

            # 4. A方计算[[masked_dL_A]]并发送给C方
            client_a.compute_masked_dL_a('C')

            # 5. C方解密[[L]]、[[masked_d_A]]和[[masked_d_B]]，并将解密后的masked_d_A发送给A，masked_d_B发送给B
            client_c.decrypt_and_distribute('A', 'B')

            # 6. A方和B方根据梯度更新模型参数
            client_a.update_params()
            client_b.update_params()
        # 获取历史损失
        self.loss_history = client_c.loss_history

        # 获取模型权重
        self.weightA = client_a.weights
        self.weightB = client_b.weights

        logger.info("训练流程完成。")

    def predict(self, XA, XB):
        pass

    def predict_proba(self, XA, XB):
        d = XA.dot(self.weightA) + XB.dot(self.weightB)
        f = 1 / (1 + np.exp(-d))
        return f

    def get_loss_history(self):
        return self.loss


class Client:
    """
    Client 基类，用于管理模型训练过程中产生的数据，并与其他 Client 实例建立连接、交换数据。

    Attributes:
        data (dict): 用于存储当前 Client 在模型训练过程中产生的数据。
        config (dict): 当前 Client 的配置参数，一般包含模型的超参数等。
        other_client (dict): 存储与当前 Client 建立连接的其它 Client 对象，键为对方名称，值为对方 Client 实例。
    """

    def __init__(self, config: dict):
        """
        初始化 Client 实例。

        Args:
            config (dict): 包含模型相关配置的字典，如超参数等。
        """
        self.data = {}
        self.config = config
        self.other_client = {}

        # 打印初始化信息
        logger.info("Client 初始化完成，config: %s", self.config)

    def connect(self, client_name: str, target_client: "Client"):
        """
        与其他 Client 建立连接，并将对方 Client 存储在本地记录中。

        Args:
            client_name (str): 对方 Client 的名称，作为标识使用。
            target_client (Client): 对方 Client 实例。
        """
        self.other_client[client_name] = target_client
        logger.info("已建立连接：当前 Client -> [%s]", client_name)

    def send_data(self, data: dict, target_client: "Client"):
        """
        向指定的 Client 发送数据。实质是将本方法参数中的 data 中的内容更新到对方的 data 中。

        Args:
            data (dict): 要发送的键值对数据。
            target_client (Client): 目标 Client 实例。
        """
        if not isinstance(data, dict):
            logger.warning("发送的数据格式应为字典，实际传入类型：%s", type(data))
            return

        target_client.data.update(data)
        logger.info("发送数据完成：%s -> %s，发送内容：%s",
                    self.__class__.__name__,
                    target_client.__class__.__name__,
                    data)


class ClientA(Client):
    """
    ClientA 通常用于存储并管理部分特征矩阵 X，以及在训练过程中维护自己的模型权重。
    继承自 Client 基类，保留了数据发送和连接等基础功能。
    """

    def __init__(self, X: np.ndarray, config: dict):
        """
        初始化 ClientA 实例，主要增加对特征矩阵和参数向量的管理。

        Args:
            X (np.ndarray): 本 Client 所拥有的特征矩阵数据。
            config (dict): 包含模型超参数和其他自定义配置的字典。
        """
        super().__init__(config)
        self.X = X
        self.n, self.m = X.shape[0], X.shape[1]

        # 初始化权重为全零向量
        self.weights = np.zeros(self.m)
        logger.info("%s 初始化完成，数据维度: %d x %d，初始权重长度: %d",
                    self.__class__.__name__, self.n, self.m, len(self.weights))

    def compute_u_a_and_squares(self, client_b_name):
        # 记录A方当前操作的日志信息
        logger.info("A方: 开始计算并加密[[u_a]]和[[u_a^2]]，准备发送给B方。")

        # 从外部数据(dt)获取公钥
        dt = self.data
        assert 'public_key' in dt, "Error: 'public_key' from C not received successfully"
        public_key = dt['public_key']
        logger.debug("A方: 成功获取到C方分发的public_key。")

        # 计算 u_a = X * weights
        u_a = self.X.dot(self.weights)
        logger.debug(f"A方: 计算得到的 u_a 形状为 {u_a.shape}。")

        # 计算 u_a^2
        u_a_square = u_a ** 2

        # 使用C方的公钥加密向量 u_a
        encrypted_u_a = np.array([public_key.encrypt(x) for x in u_a])

        # 使用C方的公钥加密向量 u_a^2
        encrypted_u_a_square = np.array([public_key.encrypt(x) for x in u_a_square])

        # 封装待发送给 B 方的数据
        data_to_B = {
            'encrypted_u_a': encrypted_u_a,
            'encrypted_u_a_square': encrypted_u_a_square
        }

        # 将加密后的数据发送给 B 方
        logger.info("A方: 加密后的[[u_a]]和[[u_a^2]]已准备就绪，开始发送给B方。")
        self.send_data(data_to_B, self.other_client[client_b_name])
        logger.info("A方: 成功将加密数据发送给B方。")

    def compute_masked_dL_a(self, client_c_name):
        # 获取encrypted_d
        dt = self.data
        assert 'encrypted_d' in dt, "Error: 'encrypted_d' from B not received successfully"
        encrypted_d = dt['encrypted_d']
        logger.debug("A方: 成功获取到B方分发的encrypted_d。")

        # 计算加密梯度encrypted_dL_a
        encrypted_dL_a = self.X.T.dot(encrypted_d)

        # 生成掩码向量，以对梯度进行本地随机扰动
        mask = np.random.rand(len(encrypted_dL_a))
        logger.debug(f"A方: 生成的随机掩码 mask 长度为 {len(mask)}。")

        # 将加密梯度与掩码相加，得到 masked_dL_a
        encrypted_masked_dL_a = encrypted_dL_a + mask
        logger.debug("A方: 已将掩码与加密梯度 dL_a 相加，得到 masked_dL_a。")

        # 将掩码保存在 A 方本地数据中，以便后续反掩码使用
        self.data.update({'mask': mask})
        logger.debug("A方: 随机掩码已保存在本地数据结构 self.data['mask']。")

        # 封装要发送给 C 方的数据
        data_to_C = {'encrypted_masked_dL_a': encrypted_masked_dL_a}

        # 发送数据给 C 方
        self.send_data(data_to_C, self.other_client[client_c_name])
        logger.info("A方: 成功将加密并掩码处理的梯度 masked_dL_a 发送给 C 方。")

    def update_params(self):
        # 使用类实例的 logger 进行日志记录
        logger.info("A方: 开始更新本地模型参数。")

        dt = self.data
        # 断言，确保掩码和解密后的梯度均已存在
        assert 'mask' in dt, "Error: 'mask' from A in step 2 not received successfully"
        assert 'masked_dL_a' in dt, "Error: 'masked_dL_a' from C in step 1 not received successfully"

        # 获取掩码和解密后的梯度
        mask = dt['mask']
        masked_dL_a = dt['masked_dL_a']

        # 恢复真实梯度 dL_a
        dL_a = masked_dL_a - mask
        logger.debug(f"A方: 恢复得到的 dL_a = {dL_a}")

        # 使用梯度更新本地模型参数
        self.weights = self.weights - self.config['lr'] * dL_a
        logger.info(f"A方: 完成模型参数更新，新的 weights = {self.weights}")

        # 控制台输出（可酌情在生产环境中关闭或记录为 debug）
        print("A weights : {}".format(self.weights))


class ClientB(Client):
    """
    ClientB 通常同时拥有特征矩阵 X 及对应标签 y，可在训练中执行局部梯度计算等任务。
    继承自 Client 基类，保留了数据发送和连接等基础功能。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, config: dict):
        """
        初始化 ClientB 实例，主要增加对特征矩阵 X、标签 y 和参数向量的管理。

        Args:
            X (np.ndarray): 特征矩阵数据。
            y (np.ndarray): 与特征矩阵对应的标签或目标值。
            config (dict): 包含模型超参数和其他自定义配置的字典。
        """
        super().__init__(config)
        self.X = X
        self.y = y
        self.n, self.m = X.shape[0], X.shape[1]

        # 初始化权重为全零向量
        self.weights = np.zeros(self.m)
        logger.info("%s 初始化完成，数据维度: %d x %d，初始权重长度: %d",
                    self.__class__.__name__, self.n, self.m, len(self.weights))

    def compute_gradient_and_loss(self, client_a_name, client_c_name):
        dt = self.data

        # 从获取公钥
        assert 'public_key' in dt, "Error: 'public_key' from C not received successfully"
        public_key = dt['public_key']
        logger.debug("A方: 成功获取到C方分发的public_key。")

        # 获取encrypted_u_a
        assert 'encrypted_u_a' in dt, "Error: 'encrypted_u_a' from A not received successfully"
        encrypted_u_a = dt['encrypted_u_a']
        logger.debug("A方: 成功获取到C方分发的encrypted_u_a。")

        # 获取encrypted_u_a_square
        dt = self.data
        assert 'encrypted_u_a_square' in dt, "Error: 'encrypted_u_a_square' from A not received successfully"
        encrypted_u_a_square = dt['encrypted_u_a_square']
        logger.debug("A方: 成功获取到C方分发的encrypted_u_a。")

        # 计算u_b
        u_b = self.X.dot(self.weights)

        # 计算u_b_square
        u_b_square = u_b ** 2

        # 使用C方的公钥加密向量 u_b
        encrypted_u_b = np.array([public_key.encrypt(x) for x in u_b])

        # 使用C方的公钥加密向量 u_a^b
        encrypted_u_b_square = np.array([public_key.encrypt(x) for x in u_b_square])

        # 计算第一项求和
        encrypted_L_1 = public_key.encrypt(np.log(2)) * self.n

        # 计算第二项
        encrypted_L_2 = np.sum((-1 / 2 * self.y) * (encrypted_u_a + encrypted_u_b))

        # 计算第三项
        encrypted_L_3 = np.sum((1 / 8 * (self.y ** 2)) * (
                encrypted_u_a_square + encrypted_u_b_square + 2 * encrypted_u_a * encrypted_u_b))

        # 计算加密损失
        encrypted_L = encrypted_L_1 + encrypted_L_2 + encrypted_L_3

        # 计算 [[d]]
        encrypted_d = public_key.encrypt(-1 / 2 * self.y) + (1 / 4 * (self.y ** 2)) * (encrypted_u_a + encrypted_u_b)
        logger.info(f"encrypted_d的形状为{encrypted_d.shape}")
        # 计算 [[dL_b]]
        encrypted_dL_b = self.X.T.dot(encrypted_d)
        logger.info(f"encrypted_dL_b的形状为{encrypted_dL_b.shape}")

        # 计算[[masked_dL_b]]
        # 生成掩码向量，以对梯度进行本地随机扰动
        mask = np.random.rand(len(encrypted_dL_b))
        logger.debug(f"A方: 生成的随机掩码 mask 长度为 {len(mask)}。")

        # 将加密梯度与掩码相加，得到 masked_dL_a
        encrypted_masked_dL_b = encrypted_dL_b + mask
        logger.debug("B方: 已将掩码与加密梯度 dL_b 相加，得到 masked_dL_b。")

        # 将掩码保存在本地数据中，以便后续反掩码使用
        self.data.update({'mask': mask})
        logger.debug("B方: 随机掩码已保存在本地数据结构 self.data['mask']。")

        data_to_A = {'encrypted_d': encrypted_d}

        data_to_C = {
            'encrypted_L': encrypted_L,
            'encrypted_masked_dL_b': encrypted_masked_dL_b
        }

        # 发送数据给 A 方
        self.send_data(data_to_A, self.other_client[client_a_name])
        logger.info("B方: 成功将encrypted_d 发送给 A 方。")

        # 发送数据给 C 方
        self.send_data(data_to_C, self.other_client[client_c_name])
        logger.info("A方: 成功将加密并掩码处理的梯度 masked_dL_a 发送给 C 方。")

    def update_params(self):
        # 使用类实例的 logger 进行日志记录
        logger.info("B方: 开始更新本地模型参数。")

        dt = self.data
        # 断言，确保掩码和解密后的梯度均已存在
        assert 'mask' in dt, "Error: 'mask' from B in step 2 not received successfully"
        assert 'masked_dL_b' in dt, "Error: 'masked_dL_b' from C in step 1 not received successfully"

        # 获取掩码和解密后的梯度
        mask = dt['mask']
        masked_dL_b = dt['masked_dL_b']

        # 恢复真实梯度 dL_b
        dL_b = masked_dL_b - mask
        logger.debug(f"B方: 恢复得到的 dL_b = {dL_b}")

        # 使用梯度更新本地模型参数
        self.weights = self.weights - self.config['lr'] * dL_b
        logger.info(f"B方: 完成模型参数更新，新的 weights = {self.weights}")

        # 控制台输出（可酌情在生产环境中关闭或记录为 debug）
        print("B weights : {}".format(self.weights))


class ClientC(Client):
    """
    ClientC 用于与其他 Client 进行加密、共享或聚合操作，可根据业务需求扩展加密、签名等功能。
    同时可在自身持有训练过程中产生的指标或中间结果数据。

    Attributes:
        A_data_shape (tuple): 用于存储来自 ClientA 的数据维度信息。
        B_data_shape (tuple): 用于存储来自 ClientB 的数据维度信息。
        public_key (Any): 存储当前 Client 的公钥，默认为 None，实际应用中可替换为具体的密钥类型。
        private_key (Any): 存储当前 Client 的私钥，默认为 None，实际应用中可替换为具体的密钥类型。
        loss_history (list): 训练过程中累计的损失函数值，用于观察模型收敛或性能变化。
    """

    def __init__(self, A_data_shape: tuple, B_data_shape: tuple, config: dict):
        """
        初始化 ClientC 实例，主要记录 ClientA 和 ClientB 的数据维度信息，
        并准备公钥和私钥以便在需要时进行加解密或签名操作。

        Args:
            A_data_shape (tuple): ClientA 的数据维度信息 (n, m)。
            B_data_shape (tuple): ClientB 的数据维度信息 (n, m)。
            config (dict): 包含模型超参数和其他自定义配置的字典。
        """
        super().__init__(config)
        self.A_data_shape = A_data_shape
        self.B_data_shape = B_data_shape
        self.public_key = None
        self.private_key = None

        # 记录训练过程中的损失
        self.loss_history = []

        logger.info("%s 初始化完成，A_data_shape=%s, B_data_shape=%s",
                    self.__class__.__name__, self.A_data_shape, self.B_data_shape)

    def generate_and_distribute_keys(self, client_a_name, client_b_name):
        """
        生成Paillier公私钥对，并分别将公钥分发给A、B方。

        Parameters
        ----------
        client_a_name : str
            在本客户端的other_client中标识A方的名称。
        client_b_name : str
            在本客户端的other_client中标识B方的名称。

        Returns
        -------
        None
            该方法无返回值，会将公钥分发给A、B方。

        Notes
        -----
        1. 生成Paillier密钥对 (public_key, private_key)。
        2. 通过 send_data 方法分别向A、B方发送public_key。
        3. 该函数只负责密钥的分发，后续对private_key的使用需要在C方本地完成。
        """
        # 记录C方的操作日志
        logger.info("C方正在生成密钥对并分发公钥给A、B方。")

        # 调用外部方法 generate_paillier_keypair 生成公钥和私钥
        self.public_key, self.private_key = SimpleHomomorphicEncryption.generate_paillier_keypair()

        # 封装需要发送给其他客户端的数据
        data_to_AB = {'public_key': self.public_key}

        # 将公钥分发给A方
        logger.info(f"向客户端 {client_a_name} 发送公钥。")
        self.send_data(data_to_AB, self.other_client[client_a_name])

        # 将公钥分发给B方
        logger.info(f"向客户端 {client_b_name} 发送公钥。")
        self.send_data(data_to_AB, self.other_client[client_b_name])

    def decrypt_and_distribute(self, client_a_name, client_b_name):
        # 使用类实例的logger，而不是全局的logger.info
        logger.info("C方: 正在解密数据，并将解密结果分别发送给A方和B方。")

        dt = self.data
        # 断言检查，确保必要的加密数据均已存在
        assert 'encrypted_L' in dt, "Error: 'encrypted_L' from B not received successfully"
        assert 'encrypted_masked_dL_b' in dt, "Error: 'encrypted_masked_dL_b' from B not received successfully"
        assert 'encrypted_masked_dL_a' in dt, "Error: 'encrypted_masked_dL_a' from A not received successfully"

        # 提取加密数据
        encrypted_L = dt['encrypted_L']
        encrypted_masked_dL_b = dt['encrypted_masked_dL_b']
        encrypted_masked_dL_a = dt['encrypted_masked_dL_a']

        # 解密总损失 L
        L = self.private_key.decrypt(encrypted_L)
        # 这里的打印最好显眼一点，方便测试时一眼看到
        print('*' * 8, L, '*' * 8)
        logger.info(f"C方: 解密后得到的损失 L = {L}")

        # 记录解密后的损失值 L
        self.loss_history.append(L)

        # 解密 masked_dL_b、masked_dL_a
        masked_dL_b = np.array([self.private_key.decrypt(x) for x in encrypted_masked_dL_b])
        masked_dL_a = np.array([self.private_key.decrypt(x) for x in encrypted_masked_dL_a])
        logger.debug("C方: 完成对 masked_dL_b、masked_dL_a 的解密。")

        # 需要发送给 A、B 方的数据
        data_to_A = {'masked_dL_a': masked_dL_a}
        data_to_B = {'masked_dL_b': masked_dL_b}

        # 发送数据给 A、B 方
        self.send_data(data_to_A, self.other_client[client_a_name])
        logger.info("C方: 解密后的 masked_dL_a 已发送给 A 方。")

        self.send_data(data_to_B, self.other_client[client_b_name])
        logger.info("C方: 解密后的 masked_dL_b 已发送给 B 方。")



