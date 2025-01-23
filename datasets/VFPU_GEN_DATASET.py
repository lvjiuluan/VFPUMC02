import os
import pandas as pd
from consts.Constants import DATASETS_PATH
from enums.DataSetsName import DataSetsName


class VFPU_GEN_DATASET:
    def __init__(self, base_dir=DATASETS_PATH):
        """
        初始化 VFPU_GEN_DATASET 类
        :param base_dir: 存放 CSV 文件的目录
        """
        self.base_dir = base_dir
        self.datasets = {}
        self._load_datasets()

    def _load_datasets(self):
        """
        自动加载所有符合命名规则的 CSV 文件，并组织成一个字典
        """
        for dataset_name in DataSetsName:
            dataset_key = dataset_name.value
            self.datasets[dataset_key] = {
                'df_A': None,
                'df_B': None,
                'y': None
            }
            # 根据文件名加载 CSV 文件
            for file_type in ['df_A', 'df_B', 'y']:
                file_name = f"VFPU_GEN_{dataset_key}_{file_type}.csv"
                file_path = os.path.join(self.base_dir, file_name)
                if os.path.exists(file_path):
                    self.datasets[dataset_key][file_type] = pd.read_csv(file_path)

    def get_dataset(self, dataset_name):
        """
        获取指定数据集的所有部分
        :param dataset_name: 数据集名称（如 'BANK', 'CREDIT'）
        :return: 包含 df_A, df_B, y 的字典
        """
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")

    def get_dataset_parts(self, dataset_name):
        """
        获取指定数据集的所有部分，并直接返回 df_A, df_B, y，方便解包
        :param dataset_name: 数据集名称（如 'BANK', 'CREDIT'）
        :return: df_A, df_B, y 的 DataFrame
        """
        if dataset_name in self.datasets:
            dataset = self.datasets[dataset_name]
            return dataset['df_A'], dataset['df_B'], dataset['y']
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")


    def get_part(self, dataset_name, part_name):
        """
        获取指定数据集的某一部分
        :param dataset_name: 数据集名称（如 'BANK', 'CREDIT'）
        :param part_name: 部分名称（如 'df_A', 'df_B', 'y'）
        :return: 指定部分的 DataFrame
        """
        if dataset_name in self.datasets:
            if part_name in self.datasets[dataset_name]:
                return self.datasets[dataset_name][part_name]
            else:
                raise ValueError(f"Part {part_name} not found in dataset {dataset_name}.")
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")

    def list_datasets(self):
        """
        列出所有可用的数据集名称
        :return: 数据集名称列表
        """
        return list(self.datasets.keys())

    def save_part(self, dataset_name, part_name, df):
        """
        保存指定数据集的某一部分到 CSV 文件
        :param dataset_name: 数据集名称（如 'BANK', 'CREDIT'）
        :param part_name: 部分名称（如 'df_A', 'df_B', 'y'）
        :param df: 要保存的 DataFrame
        """
        if dataset_name in self.datasets:
            if part_name in self.datasets[dataset_name]:
                file_name = f"VFPU_GEN_{dataset_name}_{part_name}.csv"
                file_path = os.path.join(self.base_dir, file_name)
                df.to_csv(file_path, index=False)
                self.datasets[dataset_name][part_name] = df
            else:
                raise ValueError(f"Part {part_name} not found in dataset {dataset_name}.")
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")

    def add_dataset(self, dataset_name, df_A=None, df_B=None, y=None):
        """
        添加一个新的数据集
        :param dataset_name: 数据集名称（如 'NEW_DATASET'）
        :param df_A: df_A 的 DataFrame
        :param df_B: df_B 的 DataFrame
        :param y: y 的 DataFrame
        """
        if dataset_name in self.datasets:
            raise ValueError(f"Dataset {dataset_name} already exists.")
        self.datasets[dataset_name] = {
            'df_A': df_A,
            'df_B': df_B,
            'y': y
        }
        # 保存到文件
        if df_A is not None:
            self.save_part(dataset_name, 'df_A', df_A)
        if df_B is not None:
            self.save_part(dataset_name, 'df_B', df_B)
        if y is not None:
            self.save_part(dataset_name, 'y', y)

    def delete_dataset(self, dataset_name):
        """
        删除一个数据集及其对应的文件
        :param dataset_name: 数据集名称（如 'BANK', 'CREDIT'）
        """
        if dataset_name in self.datasets:
            for part_name in ['df_A', 'df_B', 'y']:
                file_name = f"VFPU_GEN_{dataset_name}_{part_name}.csv"
                file_path = os.path.join(self.base_dir, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
            del self.datasets[dataset_name]
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")

    def filter_data(self, dataset_name, part_name, condition):
        """
        根据条件过滤数据集的某一部分
        :param dataset_name: 数据集名称（如 'BANK', 'CREDIT'）
        :param part_name: 部分名称（如 'df_A', 'df_B', 'y'）
        :param condition: 过滤条件（如 lambda x: x['column'] > 0）
        :return: 过滤后的 DataFrame
        """
        df = self.get_part(dataset_name, part_name)
        if df is not None:
            return df[condition(df)]
        else:
            raise ValueError(f"Part {part_name} in dataset {dataset_name} is empty.")

    def merge_parts(self, dataset_name, how='inner', on=None):
        """
        合并指定数据集的 df_A 和 df_B
        :param dataset_name: 数据集名称（如 'BANK', 'CREDIT'）
        :param how: 合并方式（如 'inner', 'outer', 'left', 'right'）
        :param on: 合并的键
        :return: 合并后的 DataFrame
        """
        df_A = self.get_part(dataset_name, 'df_A')
        df_B = self.get_part(dataset_name, 'df_B')
        if df_A is not None and df_B is not None:
            return pd.merge(df_A, df_B, how=how, on=on)
        else:
            raise ValueError(f"df_A or df_B in dataset {dataset_name} is empty.")