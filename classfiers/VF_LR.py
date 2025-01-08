import math
import time

import numpy as np

from .VF_BASE import VF_BASE_CLF


class VF_LR(VF_BASE_CLF):
    def __init__(self, config):
        self.config = config
        self.weightA = None
        self.weightB = None
        self.loss = []

    def fit(self, XA, XB, y):
        # XB 添加一列
        XB = np.c_[np.ones(XB.shape[0]), XB]
        Client_A = ClientA(XA, self.config)
        Client_B = ClientB(XB, y, self.config)
        Client_C = ClientC(XA.shape, XB.shape, self.config)
        for client in [Client_A, Client_B, Client_C]:
            for target_name, target_client in zip(['A', 'B', 'C'], [Client_A, Client_B, Client_C]):
                if client is not target_client:
                    client.connect(target_name, target_client)
        k = len(y) // self.config['batch_size']
        for i in range(self.config['n_iter'] * k):
            batchTimeStart = time.time()
            # print("第%d轮训练开始..." % (i+1))
            # A方向B方发送本轮batch的ID，A方设置X
            Client_A.batch_task('B')
            # B方接受A方的bathID，设置X，y
            Client_B.batch_task()
            # 第一步 C生成密钥并分发，C的task1
            Client_C.task_1('A', 'B')
            # 第二步 A、B方交换中间结果
            Client_A.task_1('B')
            Client_B.task_1('A')
            # 第三步 A、B方向C发送加密梯度
            Client_A.task_2('C')
            Client_B.task_2('C')
            # 第四步 C解密梯度并回传，计算损失
            Client_C.task_2('A', 'B')
            # 第五步 A、B方分别进行模型更新
            Client_A.task_3()
            Client_B.task_3()
            # print("第%d轮训练结束"%(i+1))
            # print("一个batch花费%f秒"%(time.time()-batchTimeStart))
            self.loss.append(Client_C.loss)
        self.weightA = Client_A.weights
        self.weightB = Client_B.weights

    def predict(self, XA, XB):
        pass

    def predict_proba(self, XA, XB):
        pass


def generate_paillier_keypair():
    return None, None


def encrypt(x):
    return x


def decrypt(x):
    return x


class Client(object):
    def __init__(self, config):
        # 模型参数
        self.config = config
        # 中间计算结果
        self.data = {}
        # 与其它节点的连接情况
        self.other_client = {}

    # 与其它方建立连接
    def connect(self, client_name, target_client):
        self.other_client[client_name] = target_client

    # 向特定方发送数据
    def send_data(self, data, target_client):
        target_client.data.update(data)


class ClientA(Client):
    def __init__(self, X, config):
        super().__init__(config)
        # 保存训练数据
        self.X = X
        self.X_copy = X
        # A方模型的参数，初始化为0
        self.weights = np.zeros(self.X.shape[1])

    # A方向B方发送本轮batch的ID，A方设置X
    def batch_task(self, client_B_name):
        ID = np.arange(len(self.X_copy))
        batchID = np.random.choice(ID, size=self.config['batch_size'])
        data_to_B = {'batchID': batchID}
        self.send_data(data_to_B, self.other_client[client_B_name])
        self.X = self.X_copy[batchID]

    # 计算XW
    def compute_z_a(self):
        z_a = np.dot(self.X, self.weights)
        return z_a

    # 加密梯度计算
    def compute_encrypted_dJ_a(self, encrypted_u):
        encrypted_dJ_a = self.X.T.dot(encrypted_u) + self.config["lambda"] * self.weights
        return encrypted_dJ_a

    # 模型参数更新
    def update_weight(self, dJ_a):
        self.weights = self.weights - self.config["lr"] * dJ_a / len(self.X)

    # 将1/4*wx加密发送给B方
    def task_1(self, client_B_name):
        dt = self.data
        assert "public_key" in dt.keys(), "Error: 'public_key' from C in step 1 not successfully received."
        public_key = dt["public_key"]
        z_a = self.compute_z_a()
        u_a = 0.25 * z_a
        z_a_square = z_a ** 2
        encrypted_u_a = np.array([encrypt(x) for x in u_a])
        encrypted_z_a_square = np.array([encrypt(x) for x in z_a_square])
        dt.update({'encrypted_u_a': encrypted_u_a})
        # 将数据发送给B
        send_to_B = {'encrypted_u_a': encrypted_u_a, 'encrypted_z_a_square': encrypted_z_a_square}
        self.send_data(send_to_B, self.other_client[client_B_name])

    # 将加密梯度发送给C方进行解密
    def task_2(self, client_C_name):
        dt = self.data
        assert "encrypted_u_b" in dt.keys(), "Error: 'encrypted_u_b' from B in step 1 not successfully received."
        encrypted_u_b = dt['encrypted_u_b']
        encrypted_u = dt['encrypted_u_a'] + encrypted_u_b
        encrypted_dJ_a = self.compute_encrypted_dJ_a(encrypted_u)
        mask = np.random.rand(len(encrypted_dJ_a))
        encrypted_masked_dJ_a = encrypted_dJ_a + mask
        dt.update({'mask': mask})
        data_to_C = {'encrypted_masked_dJ_a': encrypted_masked_dJ_a}
        self.send_data(data_to_C, self.other_client[client_C_name])

    # 模型A参数更新
    def task_3(self):
        dt = self.data
        assert "masked_dJ_a" in dt.keys(), "Error: 'masked_dJ_a' from C in step 2 not sucessfully received."
        masked_dJ_a = dt['masked_dJ_a']
        dJ_a = masked_dJ_a - dt['mask']
        self.update_weight(dJ_a)
        # print(f"A weight: {self.weights}")


class ClientB(Client):
    def __init__(self, X, y, config):
        # 调用父类构造函数
        super().__init__(config)
        # 保存数据和标签
        self.X = X
        self.y = y
        self.X_copy = X
        self.y_copy = y
        # 初始化模型参数
        self.weights = np.zeros(self.X.shape[1])

    # B方接受A方的bathID，设置X，y
    def batch_task(self):
        dt = self.data
        assert 'batchID' in dt.keys(), "Error,'batchID' from A in batch_task step not receive successfully"
        batchID = dt['batchID']
        self.X = self.X_copy[batchID]
        self.y = self.y_copy[batchID]

    # 计算XW
    def compute_z_b(self):
        z_b = self.X.dot(self.weights)
        return z_b

    # 加密计算梯度
    def compute_encrypted_dJ_b(self, encrypted_u):
        encrypted_dJ_b = self.X.T.dot(encrypted_u) + self.config['lambda'] * self.weights
        return encrypted_dJ_b

    # 模型参数更新
    def update_weights(self, dJ_b):
        self.weights = self.weights - self.config['lr'] * dJ_b / len(self.X)

    # 将1/4*wx-y+1/2发送给A
    def task_1(self, client_A_name):
        dt = self.data
        assert 'public_key' in dt.keys(), "Error: 'public_key' from C in step 1 not successfuly received."
        public_key = dt['public_key']
        z_b = self.compute_z_b()
        u_b = 0.25 * z_b - self.y + 0.25
        encrypted_u_b = np.array([encrypt(x) for x in u_b])
        dt.update({'encrypted_u_b': encrypted_u_b})
        dt.update({'z_b': z_b})
        data_to_A = {'encrypted_u_b': encrypted_u_b}
        self.send_data(data_to_A, self.other_client[client_A_name])

    # 将加密梯度发送给C方进行解密
    def task_2(self, client_C_name):
        dt = self.data
        assert "encrypted_u_a" in dt.keys(), "Error: 'encrypted_u_a' from A in step 1 not successfully received."
        encrypted_u_a = dt['encrypted_u_a']
        encrypted_u = dt['encrypted_u_b'] + encrypted_u_a
        encrypted_dJ_b = self.compute_encrypted_dJ_b(encrypted_u)
        mask = np.random.rand(len(encrypted_dJ_b))
        encrypted_masked_dJ_b = encrypted_dJ_b + mask
        dt.update({'mask': mask})
        # 下面计算损失函数
        assert "encrypted_z_a_square" in dt.keys(), "Error: 'encrypted_z_a_square' from A in step 1 not successfuly received."
        encrypted_z = 4 * encrypted_u_a + dt['z_b']
        encrypted_loss = np.sum(-0.5 * self.y * encrypted_z + 0.125 * dt['encrypted_z_a_square'] + 0.125 * dt['z_b'] * (
                encrypted_z + 4 * encrypted_u_a))
        data_to_C = {'encrypted_masked_dJ_b': encrypted_masked_dJ_b, 'encrypted_loss': encrypted_loss}
        self.send_data(data_to_C, self.other_client[client_C_name])

    # 模型B方的参数进行更新
    def task_3(self):
        dt = self.data
        assert 'masked_dJ_b' in dt.keys(), "Error: masked_dJ_b from C in step 2 not sucessfully received."
        masked_dJ_b = dt['masked_dJ_b']
        dJ_b = masked_dJ_b - dt['mask']
        self.update_weights(dJ_b)
        # print(f"B weight: {self.weights}")


class ClientC(Client):
    def __init__(self, A_d_shape, B_d_shape, config):
        super().__init__(config)
        self.A_data_shape = A_d_shape
        self.B_data_shape = B_d_shape
        self.public_key = None
        self.private_key = None
        # 保存训练过程中的损失函数
        self.loss = []

    # 分发密钥
    def task_1(self, client_A_name, client_B_name):
        public_key, private_key = generate_paillier_keypair()
        self.public_key = public_key
        self.private_key = private_key
        data_to_AB = {'public_key': public_key}
        self.send_data(data_to_AB, self.other_client[client_A_name])
        self.send_data(data_to_AB, self.other_client[client_B_name])

    # 解密梯度，回传，计算损失
    def task_2(self, client_A_name, client_B_name):
        dt = self.data
        assert "encrypted_masked_dJ_a" in dt.keys(), "Error: 'encrypted_masked_dJ_a' from A in step 2 not sucessfully receive"
        assert "encrypted_masked_dJ_b" in dt.keys(), "Error: 'encrypted_masked_dJ_b' from B in step 2 not sucessfully receive"
        encrypted_masked_dJ_a = dt['encrypted_masked_dJ_a']
        encrypted_masked_dJ_b = dt['encrypted_masked_dJ_b']
        masked_dJ_a = np.array([decrypt(x) for x in encrypted_masked_dJ_a])
        masked_dJ_b = np.array([decrypt(x) for x in encrypted_masked_dJ_b])
        assert "encrypted_loss" in dt.keys(), "Error: 'encrypted_loss' from B in step 2 not successfuly receive"
        encrypted_loss = dt['encrypted_loss']
        loss = decrypt(encrypted_loss) / self.A_data_shape[0] + math.log(2)
        # print("******loss: ", loss, "******")
        self.loss.append(loss)
        # 回传
        data_to_A = {'masked_dJ_a': masked_dJ_a}
        data_to_B = {'masked_dJ_b': masked_dJ_b}
        self.send_data(data_to_A, self.other_client[client_A_name])
        self.send_data(data_to_B, self.other_client[client_B_name])
