"""
data_prepare_script.py

该脚本旨在自动化下载和预处理数据集的过程。
脚本执行以下任务：
1. 使用wget下载数据集。
2. 对数据集进行预处理：
   2.1 处理缺失值和None值。
   2.2 将类别列编码为one-hot编码。
   2.3 归一化数值列。

使用方法：
在安装了必要库（pandas, numpy, sklearn, wget）的Python环境中运行此脚本。

作者：lvjiuluan
日期：2025年2月9日
"""

from consts.Constants import DATASETS_PATH
from utils.DatasetsPrepareUtils import download_bank_marketing_dataset

download_bank_marketing_dataset()