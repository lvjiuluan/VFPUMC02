import numpy as np

from phe import paillier

from .VF_BASE import VF_BASE_REG


class VF_LinearRegression(VF_BASE_REG):
    def __init__(self, config):
        self.config = config
        self.weightA = None
        self.weightB = None
        self.loss_history = []

    def fit(self, XA, XB, y):
        # 初始化客户端对象
        Client_A = ClientA(XA, self.config)
        Client_B = ClientB(XB, y, self.config)
        Client_C = ClientC(XA.shape, XB.shape, self.config)
        # 建立连接
        for client in [Client_A, Client_B, Client_C]:
            for target_name, target_client in zip(['A', 'B', 'C'], [Client_A, Client_B, Client_C]):
                if client is not target_client:
                    client.connect(target_name, target_client)
        # 打印连接
        for client in [Client_A, Client_B, Client_C]:
            print(client.other_client)
        # 训练流程实现
        for i in range(self.config['n_iters']):
            # 1.C创建钥匙对，分发公钥给A和B
            Client_C.task_1('A', 'B')
            # 2.1 A方计算[[u_a]] , [[L_a]]发送给B方
            Client_A.task_1('B')
            # 2.2 B方计算[[d]]发送给A, 计算[[L_b]], [[L_ab]]发给C
            Client_B.task_1('A', 'C')
            # 3.1 A方计算[[dL_a]]，将[[masked_dL_a]] 发送给C
            Client_A.task_2('C')
            # 3.2 B方计算[[dL_b]],将[[maksed_dL_b]]发送给C
            Client_B.task_2('C')
            # 3.3 C方解密[[L]]，[[masked_dL_a]]解密发送给A，[[maksed_dL_b]]发送给B
            Client_C.task_2('A', 'B')
            # 4.1 A、B方更新模型
            Client_A.task_3()
            Client_B.task_3()
            # 更新每轮的损失
            self.loss_history.append(Client_C.loss_history)
        # 更新模型权重
        self.weightA = Client_A.weights
        self.weightB = Client_B.weights

    def predict(self, XA, XB):
        return XA.dot(self.weightA) + XB.dot(self.weightB)


class Client(object):
    def __init__(self, config):
        # 模型训练过程中产生的所有数据
        self.data = {}
        self.config = config
        self.other_client = {}

    # 与其它方建立连接
    def connect(self, client_name, target_client):
        self.other_client[client_name] = target_client

    def send_data(self, data, target_client):
        target_client.data.update(data)


class ClientA(Client):
    def __init__(self, X, config):
        super().__init__(config)
        self.X = X
        self.n = X.shape[0]
        # 初始化参数
        self.weights = np.zeros(self.X.shape[1])

    # 计算u_a
    def compute_u_a(self):
        u_a = self.X.dot(self.weights)
        return u_a

    # 计算加密梯度
    def compute_encrypted_dL_a(self, encrypted_d):
        encrypted_dL_a = self.X.T.dot(encrypted_d) / self.n + self.config['lambda'] * self.weights
        return encrypted_dL_a

    # 做predict
    def predict(self, X_test):
        u_a = X_test.dot(self.weights)
        return u_a

    # 计算[[u_a]],[[L_a]]发送给B方
    def task_1(self, client_B_name):
        dt = self.data
        # 获取公钥
        assert 'public_key' in dt.keys(), "Error: 'public_key' from C in step 1 not receive successfully"
        public_key = dt['public_key']
        u_a = self.compute_u_a()
        encrypted_u_a = np.array([public_key.encrypt(x) for x in u_a])
        u_a_square = u_a ** 2
        L_a = 0.5 * np.sum(u_a_square) / self.n + 0.5 * self.config['lambda'] * np.sum(self.weights ** 2)
        encrypted_L_a = public_key.encrypt(L_a)
        data_to_B = {'encrypted_u_a': encrypted_u_a, 'encrypted_L_a': encrypted_L_a}
        self.send_data(data_to_B, self.other_client[client_B_name])

    # 计算加密梯度[[dL_a]]，加上随机数之后，发送给C
    def task_2(self, client_C_name):
        dt = self.data
        assert 'encrypted_d' in dt.keys(), "Error: 'encrypted_d' from B in step 1 not receive successfully"
        encrypted_d = dt['encrypted_d']
        encrypted_dL_a = self.compute_encrypted_dL_a(encrypted_d)
        mask = np.random.rand(len(encrypted_dL_a))
        encrypted_masked_dL_a = encrypted_dL_a + mask
        self.data.update({'mask': mask})
        data_to_C = {'encrypted_masked_dL_a': encrypted_masked_dL_a}
        self.send_data(data_to_C, self.other_client[client_C_name])

    # 获取解密后的masked梯度，减去mask，梯度下降更新
    def task_3(self):
        dt = self.data
        assert 'mask' in dt.keys(), "Error: 'mask' form A in step 2 not receive successfully"
        assert 'masked_dL_a' in dt.keys(), "Error: 'masked_dL_a' from C in step 1 not receive successfully"
        mask = dt['mask']
        masked_dL_a = dt['masked_dL_a']
        dL_a = masked_dL_a - mask
        # 注意这里的1/n
        self.weights = self.weights - self.config['lr'] * dL_a
        print("A weights : {}".format(self.weights))


class ClientB(Client):
    def __init__(self, X, y, config):
        super().__init__(config)
        self.X = X
        self.y = y
        self.weights = np.zeros(self.X.shape[1])

    # 计算u_b
    def compute_u_b(self):
        u_b = self.X.dot(self.weights)
        return u_b

    # 计算加密梯度
    def compute_encrypted_dL_b(self, encrypted_d):
        encrypted_dL_b = self.X.T.dot(encrypted_d) / self.n + self.config['lambda'] * self.weights
        return encrypted_dL_b

    # 做predict
    def predict(self, X_test):
        u_b = X_test.dot(self.weights)
        return u_b

    # 计算[[d]] 发送给A方；计算[[L_b]], [[L_ab]]，发送给C方
    def task_1(self, client_A_name, client_C_name):
        dt = self.data
        assert 'encrypted_u_a' in dt.keys(), "Error: 'encrypted_u_a' from A in step 1 not receive successfully"
        encrypted_u_a = dt['encrypted_u_a']
        u_b = self.compute_u_b()
        z_b = u_b - self.y
        z_b_square = z_b ** 2
        encrypted_d = encrypted_u_a + z_b
        data_to_A = {'encrypted_d': encrypted_d}
        self.data.update({'encrypted_d': encrypted_d})
        assert 'encrypted_L_a' in dt.keys(), "Error,'encrypted_L_a' from A in step 1 not receive successfully"
        encrypted_L_a = dt['encrypted_L_a']
        L_b = 0.5 * np.sum(z_b_square) / self.n + 0.5 * self.config['lambda'] * np.sum(self.weights ** 2)
        L_ab = np.sum(encrypted_u_a * z_b) / self.n
        encrypted_L = encrypted_L_a + L_b + L_ab
        data_to_C = {'encrypted_L': encrypted_L}
        self.send_data(data_to_A, self.other_client[client_A_name])
        self.send_data(data_to_C, self.other_client[client_C_name])

    # 计算加密梯度[[dL_b]],mask之后发给C方
    def task_2(self, client_C_name):
        dt = self.data
        assert 'encrypted_d' in dt.keys(), "Error: 'encrypted_d' from B in step 1 not receive successfully"
        encrypted_d = dt['encrypted_d']
        encrypted_dL_b = self.compute_encrypted_dL_b(encrypted_d)
        mask = np.random.rand(len(encrypted_dL_b))
        encrypted_masked_dL_b = encrypted_dL_b + mask
        self.data.update({'mask': mask})
        data_to_C = {'encrypted_masked_dL_b': encrypted_masked_dL_b}
        self.send_data(data_to_C, self.other_client[client_C_name])

    # 获取解密后的梯度，解mask，模型更新
    def task_3(self):
        dt = self.data
        assert 'mask' in dt.keys(), "Error: 'mask' form B in step 2 not receive successfully"
        assert 'masked_dL_b' in dt.keys(), "Error: 'masked_dL_b' from C in step 1 not receive successfully"
        mask = dt['mask']
        masked_dL_b = dt['masked_dL_b']
        dL_b = masked_dL_b - mask
        self.weights = self.weights - self.config['lr'] * dL_b
        print("B weights : {}".format(self.weights))


class ClientC(Client):
    def __init__(self, A_d_shape, B_d_shape, config):
        super().__init__(config)
        self.A_data_shape = A_d_shape
        self.B_data_shape = B_d_shape
        self.public_key = None
        self.private_key = None
        # 保存训练过程中的损失函数
        self.loss_history = []

    # 产生钥匙对，将公钥发送给A,B方
    def task_1(self, client_A_name, client_B_name):
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
        data_to_AB = {'public_key': self.public_key}
        self.send_data(data_to_AB, self.other_client[client_A_name])
        self.send_data(data_to_AB, self.other_client[client_B_name])

    # 解密[[L]]、[[masked_dL_a]],[[masked_dL_b]]，分别发送给A、B
    def task_2(self, client_A_name, client_B_name):
        dt = self.data
        assert 'encrypted_L' in dt.keys(), "Error: 'encrypted_L' from B in step 2 not receive successfully"
        assert 'encrypted_masked_dL_b' in dt.keys(), "Error: 'encrypted_masked_dL_b' from B in step 2 not receive successfully"
        assert 'encrypted_masked_dL_a' in dt.keys(), "Error: 'encrypted_masked_dL_a' from A in step 2 not receive successfully"
        encrypted_L = dt['encrypted_L']
        encrypted_masked_dL_b = dt['encrypted_masked_dL_b']
        encrypted_masked_dL_a = dt['encrypted_masked_dL_a']
        L = self.private_key.decrypt(encrypted_L)
        print('*' * 8, L, '*' * 8)
        self.loss_history.append(L)
        masked_dL_b = np.array([self.private_key.decrypt(x) for x in encrypted_masked_dL_b])
        masked_dL_a = np.array([self.private_key.decrypt(x) for x in encrypted_masked_dL_a])
        data_to_A = {'masked_dL_a': masked_dL_a}
        data_to_B = {'masked_dL_b': masked_dL_b}
        self.send_data(data_to_A, self.other_client[client_A_name])
        self.send_data(data_to_B, self.other_client[client_B_name])
