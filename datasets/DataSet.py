import numpy as np
import pandas as pd
import pickle
import os
import logging
from enums.SplitRatio import SplitRatio
from utils.DataProcessUtils import vertical_split
from consts.Constants import DATASETS_PATH
from enums.HideRatio import HideRatio

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataSet:
    def __init__(self, baseFileName):
        self.baseFileName = baseFileName
        self.csvFileName = f"{self.baseFileName}.csv"
        logging.info(f"正在加载数据集 {self.csvFileName}")
        self.df = pd.read_csv(os.path.join(DATASETS_PATH, self.csvFileName))
        self.y = self.df.iloc[:, -1]
        logging.info(f"数据集 {self.csvFileName} 加载完成，数据集形状为 {self.df.shape}")

    def get_data(self):
        df = self.df
        # 分离特征矩阵和标签向量
        X = df.iloc[:, :-1].values  # 除最后一列外的所有列
        y = df.iloc[:, -1].values  # 最后一列

        return X, y

    def get_train_X_y(self):
        return self.df.values.copy(), self.hidden_y.values.copy()

    def get_data_by_split_ratio(self, splitRation):
        assert isinstance(splitRation, SplitRatio), f"splitRation 必须是 SplitRatio 类型，但接收到的是 {type(splitRation)}"
        pklFileName = f"{self.baseFileName}_{splitRation.name}.pkl"
        pklFilePath = os.path.join(DATASETS_PATH, pklFileName)

        if os.path.exists(pklFilePath):
            logging.info(f"找到缓存文件 {pklFileName}，正在加载...")
            with open(pklFilePath, 'rb') as file:
                df1, df2 = pickle.load(file)
            self.dfA, self.dfB = df1, df2
            logging.info(f"数据集已从缓存文件 {pklFileName} 加载，dfA 形状: {df1.shape}, dfB 形状: {df2.shape}")
            return df1, df2, self.y
        else:
            logging.info(f"缓存文件 {pklFileName} 不存在，正在根据比例 {splitRation.value} 切分数据集...")
            split_rate = splitRation.value
            df1, df2 = vertical_split(self.df, split_rate)
            with open(pklFilePath, 'wb') as file:
                pickle.dump((df1, df2), file)
            self.dfA, self.dfB = df1, df2
            logging.info(f"数据集切分完成，dfA 形状: {df1.shape}, dfB 形状: {df2.shape}，并已保存到 {pklFileName}")
            return df1, df2, self.y

    def get_hidden_labels(self, hideRatio):
        assert isinstance(hideRatio, HideRatio), f"hideRatio 必须是 HideRatio 类型，但接收到的是 {type(hideRatio)}"
        pklFileName = f"{self.baseFileName}_hidden_{hideRatio.name}.pkl"
        pklFilePath = os.path.join(DATASETS_PATH, pklFileName)

        if os.path.exists(pklFilePath):
            logging.info(f"找到隐藏标签缓存文件 {pklFileName}，正在加载...")
            with open(pklFilePath, 'rb') as file:
                hidden_y = pickle.load(file)
            self.hidden_y = hidden_y
            logging.info(f"隐藏标签已从缓存文件 {pklFileName} 加载")
            return hidden_y

        logging.info(f"隐藏标签缓存文件 {pklFileName} 不存在，正在生成隐藏标签...")
        hide_rate = hideRatio.value
        hidden_y = self.y.copy()

        num_samples = len(hidden_y)
        num_to_hide = int(hide_rate * num_samples)
        hide_indices = np.random.choice(num_samples, num_to_hide, replace=False)

        hidden_y.iloc[hide_indices] = -1

        with open(pklFilePath, 'wb') as file:
            pickle.dump(hidden_y, file)

        self.hidden_y = hidden_y
        logging.info(f"隐藏标签生成完成，已隐藏 {num_to_hide} 个标签，并保存到 {pklFileName}")
        return hidden_y

    def datasetInfo(self):
        logging.info("正在获取数据集信息...")
        if hasattr(self, 'hidden_y') and self.hidden_y is not None:
            num_hidden = (self.hidden_y == -1).sum()
            num_rest = len(self.hidden_y) - num_hidden
            logging.info(
                f"标签y总数量为：{len(self.hidden_y)}, 隐藏标签的数量为：{num_hidden}，比例为 {round(num_hidden / len(self.hidden_y), 2)}")
            logging.info(
                f"有标签的数量为：{num_rest}, 标签为1的数量为：{(self.hidden_y == 1).sum()}, 标签为0的数量为：{(self.hidden_y == 0).sum()}")

        if hasattr(self, 'df') and self.df is not None and not self.df.empty:
            logging.info(f"数据集的形状为: {self.df.shape}")

        if hasattr(self, 'dfA') and self.dfA is not None and not self.dfA.empty:
            logging.info(f"dfA的形状为：{self.dfA.shape}")

        if hasattr(self, 'dfB') and self.dfB is not None and not self.dfB.empty:
            logging.info(f"dfB的形状为：{self.dfB.shape}")

    # 新增方法：保存数据集到文件
    def save_dataset(self, file_name):
        file_path = os.path.join(DATASETS_PATH, file_name)
        logging.info(f"正在保存数据集到 {file_path}...")
        self.df.to_csv(file_path, index=False)
        logging.info(f"数据集已保存到 {file_path}")

    # 新增方法：加载数据集
    def load_dataset(self, file_name):
        file_path = os.path.join(DATASETS_PATH, file_name)
        logging.info(f"正在从 {file_path} 加载数据集...")
        self.df = pd.read_csv(file_path)
        self.y = self.df.iloc[:, -1]
        logging.info(f"数据集加载完成，数据集形状为 {self.df.shape}")

    def count_hidden_positives(self):
        """
        获取隐藏标签中正样本的个数。

        参数:
        hidden_y (pd.Series): 包含 1, 0, -1 的标签，其中 -1 表示隐藏标签。
        y (pd.Series): 原始标签，包含 1 和 0。

        返回:
        int: 隐藏标签中正样本的个数。
        """
        # 找到 hidden_y 中标签为 -1 的索引
        hidden_indices = self.hidden_y[self.hidden_y == -1].index

        # 在 y 中查找这些索引对应的标签，并统计正样本（1）的个数
        hidden_positive_count = self.y.loc[hidden_indices].sum()

        return hidden_positive_count


class BankDataset(DataSet):
    def __init__(self):
        super().__init__("bank")


class CensusDataset(DataSet):
    def __init__(self):
        super().__init__("census")


class CreditDataset(DataSet):
    def __init__(self):
        super().__init__("credit")


def get_all_dataset():
    return [BankDataset(), CensusDataset(), CreditDataset()]
