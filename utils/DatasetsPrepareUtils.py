import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import subprocess
from consts.Constants import DATASETS_PATH
import shutil
from urllib.parse import urlparse

def prepare_updated_pollution_dataset(file_path):
    # 使用os.path.join构建文件路径
    csv_file_path = os.path.join(file_path, 'updated_pollution_dataset.csv')

    # 读取csv文件
    df = pd.read_csv(csv_file_path)

    # 将分类标签转换为数值标签
    le = LabelEncoder()
    df['Air Quality'] = le.fit_transform(df['Air Quality'])

    # 分离特征和目标变量
    X = df.drop(columns=['Air Quality'])
    y = df['Air Quality']

    # 对X进行归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 创建新的DataFrame并保存
    prepared_df = pd.DataFrame(data=X_scaled, columns=X.columns)
    prepared_df['Air Quality'] = y
    prepared_df.to_csv(os.path.join(file_path, 'updated_pollution_dataset_prepared.csv'), index=False)


def download_file(download_url: str, path: str, filename: str = None):
    """
    使用 Linux wget 下载文件到指定目录，并可指定文件名。
    如果目标文件已存在，则跳过下载。

    :param download_url: 要下载的文件链接
    :param path: 存储文件的目标目录
    :param filename: (可选) 指定保存的文件名，默认为 None，使用 wget 默认文件名
    """
    # 确保目标目录存在
    os.makedirs(path, exist_ok=True)

    # 如果未指定 filename，则从 URL 提取文件名
    if not filename:
        filename = os.path.basename(urlparse(download_url).path)

    # 计算最终文件路径
    output_path = os.path.join(path, filename)

    # 检查文件是否已存在
    if os.path.exists(output_path):
        print(f"✅ 文件已存在，跳过下载: {output_path}")
        return output_path  # 直接返回已存在的文件路径

    # 构造 wget 命令
    command = ["wget", "-c", "-O", output_path, download_url]  # `-c` 选项支持断点续传

    try:
        # 执行 wget 命令
        subprocess.run(command, check=True)
        print(f"✅ 文件已成功下载到: {output_path}")
        return output_path  # 返回下载的文件路径
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        return None  # 失败返回 None



