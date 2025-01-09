from utils.SimpleHomomorphicEncryption import SimpleHomomorphicEncryption
import logging

# 配置基础日志
# 注意：在实际生产环境中，通常不会直接在代码中调用 basicConfig，而是在主入口（如 main.py 或配置文件）中进行配置。
import numpy as np

from classfiers.VF_BASE import VF_BASE_REG

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


class VF_LinearRegression(VF_BASE_REG):
    """
    VF_LinearRegression: 一个纵向联邦线性回归器，继承自 VF_BASE_REG。

    参数:
        config (dict): 包含模型的超参数配置，支持以下键值对：
            - batch_size (int): 每次训练的批大小。
            - lr (float): 学习率，用于梯度下降。
            - lambda (float): 权重正则化因子，用于 L2 正则化。
            - n_iter (int): 训练的最大迭代次数。

    属性:
        weightA (ndarray or None): 模型的权重参数 A，初始化为 None。
        weightB (ndarray or None): 模型的权重参数 B，初始化为 None。
        loss_history (list): 记录每次迭代的损失值，用于分析训练过程。
    """

    def __init__(self, config):
        """
        初始化 VF_LinearRegression 模型。

        参数:
            config (dict): 模型的超参数配置。
        """
        # 配置模型参数
        self.config = config  # 模型相关参数，包含 batch_size, lr, lambda, n_iter 等
        self.weightA = None  # 权重 A，初始化为 None
        self.weightB = None  # 权重 B，初始化为 None
        self.loss_history = []  # 用于记录每次迭代的损失值

        # 日志记录
        logging.info("VF_LinearRegression 模型初始化完成。")
        logging.info("模型配置: %s", self.config)

    def fit(self, XA, XB, y):
        """
        联邦训练的主要流程函数。

        Parameters
        ----------
        XA : ndarray
            A方特征数据。
        XB : ndarray
            B方特征数据。
        y : ndarray
            目标标签，通常由B方持有。
        """

        logging.info("开始构建A、B、C三个客户端...")
        # 初始化客户端对象
        client_a = ClientA(XA, self.config)
        client_b = ClientB(XB, y, self.config)
        client_c = ClientC(XA.shape, XB.shape, self.config)

        # 建立客户端之间的连接
        for client, name in zip([client_a, client_b, client_c], ['A', 'B', 'C']):
            logging.info(f"正在为客户端{name}建立连接...")
            for target_name, target_client in zip(['A', 'B', 'C'], [client_a, client_b, client_c]):
                if client is not target_client:
                    client.connect(target_name, target_client)

        # 查看连接信息
        logging.info("查看各客户端的连接情况:")
        for client, name in zip([client_a, client_b, client_c], ['A', 'B', 'C']):
            logging.info(f"客户端{name}已连接: {list(client.other_client.keys())}")

        # 开始迭代训练
        logging.info("开始训练迭代流程...")
        for iteration in range(self.config['n_iters']):
            logging.info(f"========== 第 {iteration + 1} 次迭代开始 ==========")
            # 1. C方创建钥匙对，分发公钥给A和B
            client_c.generate_and_distribute_keys('A', 'B')

            # 2. A方计算[[u_a]]和[[L_a]]发送给B方
            client_a.encrypt_and_send_data_to_B('B')

            # 3. B方计算[[d]]发送给A, 计算[[L_b]]和[[L_ab]]发送给C
            client_b.encrypt_and_send_data_to_A_and_C('A', 'C')

            # 4. A方计算[[dL_a]]，将[[masked_dL_a]]发送给C
            client_a.encrypt_and_send_masked_gradient_a_to_C('C')

            # 5. B方计算[[dL_b]]，将[[masked_dL_b]]发送给C
            client_b.encrypt_and_send_masked_gradient_b_to_C('C')

            # 6. C方解密[[L]]、[[masked_dL_a]]、[[masked_dL_b]]并发送结果给A、B
            client_c.decrypt_and_distribute_results('A', 'B')

            # 7. A、B方接收解密信息后，分别更新模型参数
            client_a.receive_decrypted_data_and_update_parameters()
            client_b.receive_decrypted_data_and_update_parameters()

            logging.info(f"========== 第 {iteration + 1} 次迭代结束 ==========")

        # 获取历史损失
        self.loss_history = client_c.loss_history

        # 获取模型权重
        self.weightA = client_a.weights
        self.weightB = client_b.weights

        logging.info("训练流程完成。")

    def predict(self, XA, XB):
        """
        使用A、B两方的模型参数进行预测。

        本函数主要步骤：
        1. 接收来自 A 方 (XA) 和 B 方 (XB) 的特征矩阵；
        2. 分别将 XA 与 self.weightA 矩阵相乘，以及 XB 与 self.weightB 矩阵相乘；
        3. 将两部分结果相加得到最终的预测值。

        Parameters
        ----------
        XA : numpy.ndarray
            A方的特征矩阵，形状通常为 (n_samples, n_features_A)。
        XB : numpy.ndarray
            B方的特征矩阵，形状通常为 (n_samples, n_features_B)。

        Returns
        -------
        numpy.ndarray
            模型对输入数据的预测结果，形状通常为 (n_samples, ) 或 (n_samples, 1)。

        Notes
        -----
        - 如果是回归任务，则返回的预测结果为连续值；
        - 如果是分类任务，通常需要对输出结果进行后处理（如阈值判断、取整、softmax等）。
        - 请确认在调用此方法前，A、B两方的权重 (weightA, weightB) 已训练或加载完成。
        """

        # 使用类实例的 logger 进行日志记录
        logging.info("开始进行预测。")
        logging.debug(f"XA shape: {XA.shape}, XB shape: {XB.shape}")
        logging.debug(f"A方权重 shape: {self.weightA.shape}, B方权重 shape: {self.weightB.shape}")

        # 计算预测结果
        predictions = XA.dot(self.weightA) + XB.dot(self.weightB)

        logging.debug(f"预测结果示例（前5条）: {predictions[:5]}")
        logging.info("预测完成。")
        return predictions


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
        logging.info("Client 初始化完成，config: %s", self.config)

    def connect(self, client_name: str, target_client: "Client"):
        """
        与其他 Client 建立连接，并将对方 Client 存储在本地记录中。

        Args:
            client_name (str): 对方 Client 的名称，作为标识使用。
            target_client (Client): 对方 Client 实例。
        """
        self.other_client[client_name] = target_client
        logging.info("已建立连接：当前 Client -> [%s]", client_name)

    def send_data(self, data: dict, target_client: "Client"):
        """
        向指定的 Client 发送数据。实质是将本方法参数中的 data 中的内容更新到对方的 data 中。

        Args:
            data (dict): 要发送的键值对数据。
            target_client (Client): 目标 Client 实例。
        """
        if not isinstance(data, dict):
            logging.warning("发送的数据格式应为字典，实际传入类型：%s", type(data))
            return

        target_client.data.update(data)
        logging.info("发送数据完成：%s -> %s，发送内容：%s",
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
        logging.info("%s 初始化完成，数据维度: %d x %d，初始权重长度: %d",
                     self.__class__.__name__, self.n, self.m, len(self.weights))

    def encrypt_and_send_data_to_B(self, client_B_name):
        """
        A方计算并加密数据[[u_a]]、[[L_a]]，并将其发送给B方以便后续训练流程使用。

        本函数主要步骤：
        1. 从 `self.data` 中获取由C方分发的Paillier `public_key`；
        2. 使用 A 方本地数据 `self.X` 与模型参数 `self.weights` 计算得到向量 `u_a`；
        3. 计算损失项 `L_a`，包含回归损失和正则项；
        4. 使用 `public_key` 对 `u_a` 和 `L_a` 进行加密；
        5. 通过 `send_data` 方法，将加密后的数据发送给 B 方。

        Parameters
        ----------
        client_B_name : str
            在 `self.other_client` 中标识 B 方的名称，用于获取 B 方对象并进行发送。

        Returns
        -------
        None
            函数无返回值，直接将加密后的数据发送给 B 方。

        Raises
        ------
        AssertionError
            当 `public_key` 未正确接收到时，抛出断言异常。
        """

        # 记录A方当前操作的日志信息
        logging.info("A方: 开始计算并加密[[u_a]]和[[L_a]]，准备发送给B方。")

        # 从外部数据(dt)获取公钥
        dt = self.data
        assert 'public_key' in dt, "Error: 'public_key' from C in step 2 not received successfully"
        public_key = dt['public_key']
        logging.debug("A方: 成功获取到C方分发的public_key。")

        # 计算 u_a = X * weights
        u_a = self.X.dot(self.weights)
        logging.debug(f"A方: 计算得到的 u_a 形状为 {u_a.shape}。")

        # 使用C方的公钥加密向量 u_a
        # 注意：如需进一步优化，可视情况批量加密或采用更高效的加解密策略
        encrypted_u_a = np.array([public_key.encrypt(x) for x in u_a])

        # 计算损失 L_a = 0.5 * ∑(u_a^2) / n + 0.5 * λ * ∑(weights^2)
        u_a_square = u_a ** 2
        L_a = 0.5 * np.sum(u_a_square) / self.n + 0.5 * self.config['lambda'] * np.sum(self.weights ** 2)
        logging.debug(f"A方: 计算得到的 L_a = {L_a}")

        # 对 L_a 进行加密
        encrypted_L_a = public_key.encrypt(L_a)

        # 封装待发送给 B 方的数据
        data_to_B = {
            'encrypted_u_a': encrypted_u_a,
            'encrypted_L_a': encrypted_L_a
        }

        # 将加密后的数据发送给 B 方
        logging.info("A方: 加密后的[[u_a]]和[[L_a]]已准备就绪，开始发送给B方。")
        self.send_data(data_to_B, self.other_client[client_B_name])
        logging.info("A方: 成功将加密数据发送给B方。")

    def encrypt_and_send_masked_gradient_a_to_C(self, client_C_name):
        """
        A方计算[[dL_a]]，在本地对其进行掩码处理 (mask)，并将[[masked_dL_a]]发送给C方。

        本函数主要步骤：
        1. 从 self.data 中获取由 B 方发送的 `encrypted_d`；
        2. 使用 X.T * encrypted_d / n + λ * weights 计算梯度 `encrypted_dL_a`；
        3. 生成随机掩码向量 `mask`，并将其与 `encrypted_dL_a` 相加得到 `encrypted_masked_dL_a`；
        4. 将生成的 `encrypted_masked_dL_a` 发送给 C 方，以便后续进行解密操作。

        Parameters
        ----------
        client_C_name : str
            在 `self.other_client` 中标识 C 方的名称，用于获取 C 方对象并进行发送。

        Returns
        -------
        None
            函数无返回值，直接将加密并掩码处理的梯度发送给 C 方。

        Raises
        ------
        AssertionError
            当从 B 方未能正确接收到 `encrypted_d` 时，会抛出断言异常。

        Notes
        -----
        - `mask` 存在于 A 方本地的 self.data 中，后续C方解密后会把解密值返回给 A，再配合此 `mask` 恢复真实梯度。
        - 如果加密方式或同态运算要求更严格，可进一步对 `mask` 也进行安全处理。
        """

        # 记录A方当前步骤的日志信息
        logging.info("A方: 开始计算并发送[[masked_dL_a]]给C方。")

        # 从外部数据 dt 中获取 B 方发送的加密差值 encrypted_d
        dt = self.data
        assert 'encrypted_d' in dt, "Error: 'encrypted_d' from B not received successfully"
        encrypted_d = dt['encrypted_d']
        logging.debug("A方: 成功从 B 方获取加密差值 encrypted_d。")

        # 这里直接使用线性回归的梯度公式（或其他您需要的公式）计算加密梯度 dL_a
        # dL_a = (1/n) * X.T * d + λ * w
        encrypted_dL_a = self.X.T.dot(encrypted_d) / self.n + self.config['lambda'] * self.weights
        logging.debug("A方: 完成对加密梯度 dL_a 的计算。")

        # 生成掩码向量，以对梯度进行本地随机扰动
        mask = np.random.rand(len(encrypted_dL_a))
        logging.debug(f"A方: 生成的随机掩码 mask 长度为 {len(mask)}。")

        # 将加密梯度与掩码相加，得到 masked_dL_a
        encrypted_masked_dL_a = encrypted_dL_a + mask
        logging.debug("A方: 已将掩码与加密梯度 dL_a 相加，得到 masked_dL_a。")

        # 将掩码保存在 A 方本地数据中，以便后续反掩码使用
        self.data.update({'mask': mask})
        logging.debug("A方: 随机掩码已保存在本地数据结构 self.data['mask']。")

        # 封装要发送给 C 方的数据
        data_to_C = {'encrypted_masked_dL_a': encrypted_masked_dL_a}

        # 发送数据给 C 方
        self.send_data(data_to_C, self.other_client[client_C_name])
        logging.info("A方: 成功将加密并掩码处理的梯度 masked_dL_a 发送给 C 方。")

    def receive_decrypted_data_and_update_parameters(self):
        """
        A方在接收到C方解密后的梯度或其他必要信息后，更新本地模型参数。

        本函数主要步骤：
        1. 从 self.data 中获取之前在本地保存的掩码 `mask` 和 C 方返回的解密后梯度 `masked_dL_a`；
        2. 通过 dL_a = masked_dL_a - mask 恢复真实梯度；
        3. 使用学习率 lr 对 A 方的本地模型参数 self.weights 进行更新。

        Raises
        ------
        AssertionError
            - 如果在 self.data 中未找到 'mask' 或 'masked_dL_a' 时，会抛出断言异常。

        Notes
        -----
        - 若需要进一步的安全或隐私防护措施，可在使用梯度前进行额外检验或随机检测。
        - 更新后的模型参数可在下次迭代中被使用，也可持久化保存。
        """

        # 使用类实例的 logger 进行日志记录
        logging.info("A方: 开始更新本地模型参数。")

        dt = self.data
        # 断言，确保掩码和解密后的梯度均已存在
        assert 'mask' in dt, "Error: 'mask' from A in step 2 not received successfully"
        assert 'masked_dL_a' in dt, "Error: 'masked_dL_a' from C in step 1 not received successfully"

        # 获取掩码和解密后的梯度
        mask = dt['mask']
        masked_dL_a = dt['masked_dL_a']

        # 恢复真实梯度 dL_a
        dL_a = masked_dL_a - mask
        logging.debug(f"A方: 恢复得到的 dL_a = {dL_a}")

        # 使用梯度更新本地模型参数
        self.weights = self.weights - self.config['lr'] * dL_a
        logging.info(f"A方: 完成模型参数更新，新的 weights = {self.weights}")

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
        logging.info("%s 初始化完成，数据维度: %d x %d，初始权重长度: %d",
                     self.__class__.__name__, self.n, self.m, len(self.weights))

    def encrypt_and_send_data_to_A_and_C(self, client_A_name, client_C_name):
        """
        B方计算加密的差值向量 [[d]] 并发送给A方，
        同时计算并加密损失项 [[L_b]] 与交互项 [[L_ab]]，发送给C方。

        本函数主要步骤：
        1. 从 self.data 中获取由A方发送的加密向量 `encrypted_u_a` 以及加密损失 `encrypted_L_a`；
        2. 使用本地数据 X 和模型参数 weights 计算向量 `u_b` 与残差向量 `z_b = u_b - y`；
        3. 根据 `encrypted_u_a` 与 `z_b`，计算 `encrypted_d = encrypted_u_a + z_b` 并发送给 A 方；
        4. 计算 B 方本地损失 `L_b` 以及交互项 `L_ab`，然后将其与 `encrypted_L_a` 相加得到 `encrypted_L` 发送给 C 方；
        5. 将部分中间结果（如 `encrypted_d`）保存回 self.data，供后续流程使用。

        Parameters
        ----------
        client_A_name : str
            在 `self.other_client` 中标识 A 方的名称，用于获取 A 方对象并进行发送。
        client_C_name : str
            在 `self.other_client` 中标识 C 方的名称，用于获取 C 方对象并进行发送。

        Returns
        -------
        None
            函数无返回值，直接将计算及加密后的数据发送给 A 方与 C 方。

        Raises
        ------
        AssertionError
            当从 A 方未能正确接收到 `encrypted_u_a` 或 `encrypted_L_a` 时，会抛出断言异常。

        Notes
        -----
        - `encrypted_u_a` 与 `encrypted_L_a` 由 A 方在前一步发送，必须先在 self.data 中获得；
        - 若要进一步对加密效率进行优化，可考虑批量操作或其他安全多方计算方案。
        """

        # 记录B方当前步骤的日志信息
        logging.info(
            "B方: 开始计算加密的差值向量[[d]]并发送给A方，同时将[[L_b]]和[[L_ab]]加密后发送给C方。"
        )

        # 从外部数据(dt)获取加密向量，以及检查关键字段是否存在
        dt = self.data
        assert 'encrypted_u_a' in dt, "Error: 'encrypted_u_a' from A not received successfully"
        encrypted_u_a = dt['encrypted_u_a']
        logging.debug("B方: 成功获取到 A 方的加密向量 encrypted_u_a。")

        # 计算 B 方本地的 u_b = X * weights
        u_b = self.X.dot(self.weights)
        logging.debug(f"B方: 计算得到的 u_b 形状为 {u_b.shape}。")

        # 计算残差 z_b = u_b - y
        z_b = u_b - self.y
        z_b_square = z_b ** 2

        # 计算加密的 d = encrypted_u_a + z_b (z_b 还未加密，这里简单相加)
        # 这里假设 encrypted_u_a 支持与明文 z_b 相加。若需要加密 z_b，则需要先加密或换用同态运算。
        encrypted_d = encrypted_u_a + z_b

        # 将加密后的 d 发送给 A 方
        data_to_A = {'encrypted_d': encrypted_d}
        self.data.update({'encrypted_d': encrypted_d})  # 将中间结果保存以备后续使用
        logging.info("B方: 计算并封装了加密向量 d，准备发送给 A 方。")

        # 验证从 A 方获取的加密损失 encrypted_L_a
        assert 'encrypted_L_a' in dt, "Error: 'encrypted_L_a' from A not received successfully"
        encrypted_L_a = dt['encrypted_L_a']
        logging.debug("B方: 成功获取到 A 方的加密损失 encrypted_L_a。")

        # 计算 B 方本地损失 L_b
        L_b = 0.5 * np.sum(z_b_square) / self.n + 0.5 * self.config['lambda'] * np.sum(self.weights ** 2)
        logging.debug(f"B方: 计算得到的 L_b = {L_b}")

        # 计算交互项 L_ab = ∑(encrypted_u_a * z_b) / n
        # 注意 encrypted_u_a 和 z_b 的运算方式，以及同态加法/乘法是否正确。
        L_ab = np.sum(encrypted_u_a * z_b) / self.n
        logging.debug(f"B方: 计算得到的 L_ab (同态加密场景下为加密/明文混合表达) = {L_ab}")

        # 计算加密的 L = encrypted_L_a + L_b + L_ab
        # 其中 encrypted_L_a 可视为同态加密的数值，L_b 和 L_ab 是明文，直接相加表示对加密结果的同态运算。
        encrypted_L = encrypted_L_a + L_b + L_ab

        # 将加密后的整体损失加和发送给 C 方
        data_to_C = {'encrypted_L': encrypted_L}
        logging.info("B方: 计算并封装了加密的总体损失项 L，准备发送给 C 方。")

        # 分别发送数据给 A 方和 C 方
        self.send_data(data_to_A, self.other_client[client_A_name])
        logging.info("B方: 已成功将加密向量 d 发送给 A 方。")

        self.send_data(data_to_C, self.other_client[client_C_name])
        logging.info("B方: 已成功将加密的总体损失 L 发送给 C 方。")

    def encrypt_and_send_masked_gradient_b_to_C(self, client_C_name):
        """
        B方计算[[dL_b]]，在本地进行掩码处理 (mask)，并将[[masked_dL_b]]发送给C方。

        本函数主要步骤：
        1. 从 `self.data` 中获取由 B 方（或其他组件）发送的加密差值 `encrypted_d`；
        2. 通过本地计算公式（如线性回归的梯度）：`X.T.dot(encrypted_d) / n + λ * weights` 得到加密梯度 `encrypted_dL_b`；
        3. 生成随机掩码向量 `mask`，并将其与 `encrypted_dL_b` 相加得到 `encrypted_masked_dL_b`；
        4. 将生成的 `encrypted_masked_dL_b` 发送给 C 方，以便后续解密或聚合操作。

        Parameters
        ----------
        client_C_name : str
            在 `self.other_client` 中标识 C 方的名称，用于获取 C 方对象并进行发送。

        Returns
        -------
        None
            函数无返回值，直接将加密并掩码处理的梯度发送给 C 方。

        Raises
        ------
        AssertionError
            当从相关方未能正确接收到 `encrypted_d` 时，会抛出断言异常。

        Notes
        -----
        - `mask` 存在于 B 方本地的 `self.data` 中，以用于后续在解密或反掩码阶段恢复真实梯度。
        - 若对安全性有更高要求，可以进一步对 `mask` 也采用安全多方计算或同态加密。
        """

        # 使用类实例的 logger，而非直接调用 logging.info，保证日志记录更灵活
        logging.info("B方: 开始计算并发送[[masked_dL_b]]给C方。")

        # 从外部数据 dt 中获取加密差值 encrypted_d
        dt = self.data
        assert 'encrypted_d' in dt, "Error: 'encrypted_d' from B not received successfully"
        encrypted_d = dt['encrypted_d']
        logging.debug("B方: 成功获取到加密差值 encrypted_d。")

        # 计算加密的梯度 dL_b
        # dL_b = (X^T * encrypted_d) / n + λ * weights
        encrypted_dL_b = self.X.T.dot(encrypted_d) / self.n + self.config['lambda'] * self.weights
        logging.debug("B方: 完成加密梯度 dL_b 的计算。")

        # 生成随机掩码并与加密梯度相加
        mask = np.random.rand(len(encrypted_dL_b))
        encrypted_masked_dL_b = encrypted_dL_b + mask
        logging.debug("B方: 已将随机掩码与加密梯度 dL_b 相加，得到 masked_dL_b。")

        # 将随机掩码保存到 B 方本地，以便后续解密时使用
        self.data.update({'mask': mask})
        logging.debug("B方: 掩码已保存在 self.data['mask'] 以备后续使用。")

        # 封装要发送给 C 方的数据
        data_to_C = {'encrypted_masked_dL_b': encrypted_masked_dL_b}

        # 发送数据给 C 方
        self.send_data(data_to_C, self.other_client[client_C_name])
        logging.info("B方: 成功将加密并掩码处理的梯度 masked_dL_b 发送给 C 方。")

    def receive_decrypted_data_and_update_parameters(self):
        """
        B方在接收到C方解密后的梯度或其他必要信息后，更新本地模型参数。

        本函数主要步骤：
        1. 从 self.data 中获取之前在本地保存的掩码 `mask` 和 C 方返回的解密后梯度 `masked_dL_b`；
        2. 通过 dL_b = masked_dL_b - mask 恢复真实梯度；
        3. 使用学习率 lr 对 B 方的本地模型参数 self.weights 进行更新。

        Raises
        ------
        AssertionError
            - 如果在 self.data 中未找到 'mask' 或 'masked_dL_b' 时，会抛出断言异常。

        Notes
        -----
        - 若需要进一步的安全或隐私防护措施，可在使用梯度前进行额外检验或随机检测。
        - 更新后的模型参数可在下次迭代中被使用，也可持久化保存。
        """

        # 使用类实例的 logger 进行日志记录
        logging.info("B方: 开始更新本地模型参数。")

        dt = self.data
        # 断言，确保掩码和解密后的梯度均已存在
        assert 'mask' in dt, "Error: 'mask' from B in step 2 not received successfully"
        assert 'masked_dL_b' in dt, "Error: 'masked_dL_b' from C in step 1 not received successfully"

        # 获取掩码和解密后的梯度
        mask = dt['mask']
        masked_dL_b = dt['masked_dL_b']

        # 恢复真实梯度 dL_b
        dL_b = masked_dL_b - mask
        logging.debug(f"B方: 恢复得到的 dL_b = {dL_b}")

        # 使用梯度更新本地模型参数
        self.weights = self.weights - self.config['lr'] * dL_b
        logging.info(f"B方: 完成模型参数更新，新的 weights = {self.weights}")

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

        logging.info("%s 初始化完成，A_data_shape=%s, B_data_shape=%s",
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
        logging.info("C方正在生成密钥对并分发公钥给A、B方。")

        # 调用外部方法 generate_paillier_keypair 生成公钥和私钥
        self.public_key, self.private_key = SimpleHomomorphicEncryption.generate_paillier_keypair()

        # 封装需要发送给其他客户端的数据
        data_to_AB = {'public_key': self.public_key}

        # 将公钥分发给A方
        logging.info(f"向客户端 {client_a_name} 发送公钥。")
        self.send_data(data_to_AB, self.other_client[client_a_name])

        # 将公钥分发给B方
        logging.info(f"向客户端 {client_b_name} 发送公钥。")
        self.send_data(data_to_AB, self.other_client[client_b_name])

    def decrypt_and_distribute_results(self, client_A_name, client_B_name):
        """
        C方解密来自A、B的所有加密数据，比如解密[[L]]、[[masked_dL_a]]、[[masked_dL_b]]等，
        并将必要的解密结果分别发送给A方和B方。

        本函数主要步骤：
        1. 从 self.data 中获取加密的总损失 encrypted_L、加密梯度 encrypted_masked_dL_b、以及加密梯度 encrypted_masked_dL_a；
        2. 使用 C 方本地的私钥 self.private_key 对上述加密数据进行解密，得到明文 L、masked_dL_b、masked_dL_a；
        3. 记录解密后的损失值 L（如追加到 self.loss_history）；
        4. 将解密后的 masked_dL_a 和 masked_dL_b 分别发送给 A 方和 B 方。

        Parameters
        ----------
        client_A_name : str
            在 `self.other_client` 中标识 A 方的名称，用于将解密后的 masked_dL_a 发送给 A 方。
        client_B_name : str
            在 `self.other_client` 中标识 B 方的名称，用于将解密后的 masked_dL_b 发送给 B 方。

        Returns
        -------
        None
            本函数无返回值，仅对解密后的数据进行分发。

        Raises
        ------
        AssertionError
            当从 B 或 A 未能正确接收到所需加密数据 (encrypted_L、encrypted_masked_dL_b、encrypted_masked_dL_a) 时，抛出异常。

        Notes
        -----
        - 这里使用了 self.private_key.decrypt(...) 对加密数据进行解密。根据安全策略，私钥应当在 C 方安全管理。
        - 如果在生产环境或更复杂的隐私场景中，需要考虑对解密后的打印或日志进行脱敏或访问控制。
        """

        # 使用类实例的logger，而不是全局的logging.info
        logging.info("C方: 正在解密数据，并将解密结果分别发送给A方和B方。")

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
        logging.info(f"C方: 解密后得到的损失 L = {L}")

        # 记录解密后的损失值 L
        self.loss_history.append(L)

        # 解密 masked_dL_b、masked_dL_a
        masked_dL_b = np.array([self.private_key.decrypt(x) for x in encrypted_masked_dL_b])
        masked_dL_a = np.array([self.private_key.decrypt(x) for x in encrypted_masked_dL_a])
        logging.debug("C方: 完成对 masked_dL_b、masked_dL_a 的解密。")

        # 需要发送给 A、B 方的数据
        data_to_A = {'masked_dL_a': masked_dL_a}
        data_to_B = {'masked_dL_b': masked_dL_b}

        # 发送数据给 A、B 方
        self.send_data(data_to_A, self.other_client[client_A_name])
        logging.info("C方: 解密后的 masked_dL_a 已发送给 A 方。")

        self.send_data(data_to_B, self.other_client[client_B_name])
        logging.info("C方: 解密后的 masked_dL_b 已发送给 B 方。")
