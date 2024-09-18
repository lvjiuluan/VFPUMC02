import numpy as np
import pandas as pd
import pickle
from enums.SplitRatio import SplitRatio
from utils.DataProcessUtils import vertical_split
from consts.Constants import DATASETS_PATH
import os
from enums.HideRatio import HideRatio


class DataSet:
    def __init__(self, baseFileName):
        self.baseFileName = baseFileName
        self.csvFileName = f"{self.baseFileName}.csv"
        self.df = pd.read_csv(os.path.join(DATASETS_PATH, self.csvFileName))
        self.y = self.df.iloc[:, -1]

    def get_data_by_split_ratio(self, splitRation):
        assert isinstance(splitRation, SplitRatio), f"splitRation 必须是 SplitRatio 类型，但接收到的是 {type(splitRation)}"
        pklFileName = f"{self.baseFileName}_{splitRation.name}.pkl"
        pklFilePath = os.path.join(DATASETS_PATH, pklFileName)
        if os.path.exists(pklFilePath):
            with open(pklFilePath, 'rb') as file:
                df1, df2 = pickle.load(file)
            self.dfA, self.dfB = df1, df2
            return df1, df2, self.y
        else:
            split_rate = splitRation.value
            df1, df2 = vertical_split(self.df, split_rate)
            with open(pklFilePath, 'wb') as file:
                pickle.dump((df1, df2), file)
            self.dfA, self.dfB = df1, df2
            return df1, df2, self.y

    def get_hidden_labels(self, hideRatio):
        # 断言判断 hideRatio 必须是 HideRatio 类型
        assert isinstance(hideRatio, HideRatio), f"hideRatio 必须是 HideRatio 类型，但接收到的是 {type(hideRatio)}"

        # 生成文件名
        pklFileName = f"{self.baseFileName}_hidden_{hideRatio.name}.pkl"
        pklFilePath = os.path.join(DATASETS_PATH, pklFileName)

        # 如果文件已经存在，直接加载并返回
        if os.path.exists(pklFilePath):
            with open(pklFilePath, 'rb') as file:
                hidden_y = pickle.load(file)
            self.hidden_y = hidden_y
            return hidden_y

        # 获取隐藏比例
        hide_rate = hideRatio.value

        # 复制原始标签
        hidden_y = self.y.copy()

        # 随机选择需要隐藏的索引
        num_samples = len(hidden_y)
        num_to_hide = int(hide_rate * num_samples)
        hide_indices = np.random.choice(num_samples, num_to_hide, replace=False)

        # 将选中的索引对应的标签设为 -1
        hidden_y.iloc[hide_indices] = -1

        # 将隐藏后的标签保存到文件
        with open(pklFilePath, 'wb') as file:
            pickle.dump(hidden_y, file)

        self.hidden_y = hidden_y
        return hidden_y

    def datasetInfo(self):
        num_hidden = (self.hidden_y == -1).sum()
        num_rest = len(self.hidden_y) - num_hidden
        print(f"标签y总数量为：{len(self.hidden_y)}, 隐藏标签的数量为：{num_hidden}，比例为{round(num_hidden / len(self.hidden_y), 2)}")
        print(f"有标签为的数量为：{num_rest}, 标签为1的数量为：{(self.hidden_y == 1).sum()}, 标签为0的数量为：{(self.hidden_y == 0).sum()}")
        print(f"数据集的形状为: {self.df.shape}，dfA的形状为：{self.dfA.shape}, dfB的形状为：{self.dfB.shape}, ")


class BankDataset(DataSet):
    def __init__(self):
        super().__init__("bank")

class CensusDataset(DataSet):
    def __init__(self):
        super().__init__("census")