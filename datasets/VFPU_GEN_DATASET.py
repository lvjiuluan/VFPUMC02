import os
import re
import pandas as pd
from typing import List, Dict, Callable, Union
from consts.Constants import DATASETS_PATH


class VFPU_GEN_DATASET:
    """
    用于管理当前目录下命名为VFPU_GEN_开头的CSV数据集。
    按照 VFPU_GEN_{DATASET_NAME}_{SUBSET_NAME}.csv 的格式识别并加载数据。
    例如：
        VFPU_GEN_BANK_df_A.csv
        VFPU_GEN_BANK_df_B.csv
        VFPU_GEN_BANK_y.csv
        VFPU_GEN_CREDIT_df_A.csv
        ...
    """

    def __init__(self, directory: str = DATASETS_PATH, file_pattern: str = r"^VFPU_GEN_(.+)_(.+)\.csv$"):
        """
        参数：
            directory: 要扫描的目录，默认为当前目录(".")。
            file_pattern: 用正则表达式匹配文件名，可根据需要调整。
                          默认匹配 VFPU_GEN_ + 任意字符 + '_' + 任意字符 + '.csv'
        """
        self.directory = directory
        self.file_pattern = file_pattern
        self.datasets = self._scan_and_load()

    def _scan_and_load(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        扫描self.directory目录下所有CSV文件，匹配正则表达式后，
        以 {dataset_name: {subset_name: dataframe}} 的方式返回加载后的DataFrame字典。
        """
        datasets = {}
        pattern = re.compile(self.file_pattern, re.IGNORECASE)

        for file_name in os.listdir(self.directory):
            if file_name.lower().endswith(".csv"):
                match = pattern.match(file_name)
                if match:
                    dataset_name, subset_name = match.groups()
                    full_path = os.path.join(self.directory, file_name)

                    df = pd.read_csv(full_path)

                    if dataset_name not in datasets:
                        datasets[dataset_name] = {}
                    datasets[dataset_name][subset_name] = df
        return datasets

    # ------------------------------------------------------------------------
    # 1. 列表或查询相关的方法
    # ------------------------------------------------------------------------
    def list_datasets(self) -> List[str]:
        """
        返回所有检测到的数据集名称（dataset_name）。
        """
        return list(self.datasets.keys())

    def list_subsets(self, dataset_name: str) -> List[str]:
        """
        返回指定数据集下所有的子集名称（如 'df_A', 'df_B', 'y' 等）。
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集 '{dataset_name}' 不存在。")
        return list(self.datasets[dataset_name].keys())

    def get_dataframe(self, dataset_name: str, subset_name: str) -> pd.DataFrame:
        """
        返回指定数据集中指定子集的DataFrame。
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集 '{dataset_name}' 不存在。")
        if subset_name not in self.datasets[dataset_name]:
            raise ValueError(f"数据集 '{dataset_name}' 中 '{subset_name}' 不存在。")
        return self.datasets[dataset_name][subset_name]

    def info(self):
        """
        打印当前所有数据集及其子集的基本信息。
        """
        print("当前加载的数据集信息：")
        for ds_name, subsets in self.datasets.items():
            print(f"  数据集: {ds_name}")
            for sb_name, df in subsets.items():
                print(f"    子集: {sb_name} | 形状: {df.shape}")
        print("\n")

    # ------------------------------------------------------------------------
    # 2. 数据合并 / 操作 相关的方法
    # ------------------------------------------------------------------------
    def combine_subsets(
            self,
            dataset_name: str,
            subset_names: List[str],
            axis: int = 0,
            join: str = "outer",
            ignore_index: bool = True
    ) -> pd.DataFrame:
        """
        合并同一个数据集下多个子集的DataFrame。
        常见用法：
            - axis=0 表示纵向堆叠（append）
            - axis=1 表示横向合并（join on index）

        参数：
            dataset_name: 数据集名称
            subset_names: 要合并的子集名称列表，例如 ['df_A', 'df_B']
            axis: 合并方向，0为纵向，1为横向
            join: 如何进行合并，'outer' 或 'inner' 等
            ignore_index: 是否忽略索引（多用于axis=0时）

        返回：
            合并后的DataFrame（不会直接存回datasets中，除非你自己赋值）
        """
        df_list = [self.get_dataframe(dataset_name, sb) for sb in subset_names]
        combined_df = pd.concat(df_list, axis=axis, join=join, ignore_index=ignore_index)
        return combined_df

    def merge_with_target(
            self,
            dataset_name: str,
            feature_subset_name: str,
            target_subset_name: str,
            on: Union[str, List[str]],
            how: str = "left"
    ) -> pd.DataFrame:
        """
        将同一个数据集下的特征表（feature_subset）和目标表（target_subset）基于某个
        公共列（或多列）进行merge。

        参数：
            dataset_name: 数据集名称
            feature_subset_name: 特征DataFrame的subset名称（如 'df_A'）
            target_subset_name: 目标DataFrame的subset名称（如 'y'）
            on: merge的键
            how: merge方式，'left'/'right'/'inner'/'outer'

        返回：
            合并后的DataFrame
        """
        feature_df = self.get_dataframe(dataset_name, feature_subset_name)
        target_df = self.get_dataframe(dataset_name, target_subset_name)
        merged_df = pd.merge(feature_df, target_df, on=on, how=how)
        return merged_df

    # ------------------------------------------------------------------------
    # 3. 常用数据预处理相关的方法（可按需扩展）
    # ------------------------------------------------------------------------
    def apply_function(
            self,
            dataset_name: str,
            subset_name: str,
            func: Callable[[pd.DataFrame], pd.DataFrame]
    ):
        """
        对指定数据集的指定子集执行一个自定义函数，返回处理后的DataFrame并更新到内存。

        参数：
            dataset_name: 数据集名称
            subset_name: 子集名称
            func: 接受一个DataFrame并返回处理后DataFrame的函数
        """
        df = self.get_dataframe(dataset_name, subset_name)
        new_df = func(df)
        self.datasets[dataset_name][subset_name] = new_df  # 更新到内存

    def fillna(self, dataset_name: str, subset_name: str, value=0):
        """
        对指定DataFrame做简单的填充缺失值操作。
        """
        df = self.get_dataframe(dataset_name, subset_name)
        df_filled = df.fillna(value)
        self.datasets[dataset_name][subset_name] = df_filled

    def drop_duplicates(self, dataset_name: str, subset_name: str):
        """
        对指定DataFrame删除重复行。
        """
        df = self.get_dataframe(dataset_name, subset_name)
        df_dedup = df.drop_duplicates()
        self.datasets[dataset_name][subset_name] = df_dedup

    def get_summary(self, dataset_name: str, subset_name: str) -> pd.DataFrame:
        """
        返回DataFrame的基本统计信息（如describe()结果）。
        """
        df = self.get_dataframe(dataset_name, subset_name)
        return df.describe()

    # ------------------------------------------------------------------------
    # 4. 模型训练相关的简便方法（可根据需要扩充）
    # ------------------------------------------------------------------------
    def to_xy(
            self,
            dataset_name: str,
            feature_subset_name: str,
            target_column: str
    ) -> (pd.DataFrame, pd.Series):
        """
        将指定数据集下的某个子集，拆分为特征X和目标y，用于后续训练模型。

        参数：
            dataset_name: 数据集名称
            feature_subset_name: 特征DataFrame子集名称
            target_column: 目标列的名称

        返回：
            (X, y) - X是特征DataFrame, y是Series
        """
        df = self.get_dataframe(dataset_name, feature_subset_name)
        if target_column not in df.columns:
            raise ValueError(f"指定的目标列 '{target_column}' 在DataFrame中不存在。")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    # ------------------------------------------------------------------------
    # 5. 数据保存相关的方法
    # ------------------------------------------------------------------------
    def save_subset(
            self,
            dataset_name: str,
            subset_name: str,
            save_dir: str = None,
            index: bool = False
    ):
        """
        将指定数据集的指定子集保存为CSV文件到save_dir目录下（默认保存到初始化目录）。
        """
        if save_dir is None:
            save_dir = self.directory

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_name = f"VFPU_GEN_{dataset_name}_{subset_name}.csv"
        save_path = os.path.join(save_dir, file_name)

        df = self.get_dataframe(dataset_name, subset_name)
        df.to_csv(save_path, index=index)
        print(f"文件已保存至: {save_path}")

    def save_all(self, save_dir: str = None, index: bool = False):
        """
        将加载的所有DataFrame全部以它们原本的命名规则保存到指定目录下（或初始化的目录）。
        """
        if save_dir is None:
            save_dir = self.directory

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for dataset_name, subsets in self.datasets.items():
            for subset_name, df in subsets.items():
                file_name = f"VFPU_GEN_{dataset_name}_{subset_name}.csv"
                save_path = os.path.join(save_dir, file_name)
                df.to_csv(save_path, index=index)
                print(f"[保存] {dataset_name} - {subset_name} -> {save_path}")
