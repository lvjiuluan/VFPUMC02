import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import subprocess
import kagglehub
from consts.Constants import DATASETS_PATH
import shutil

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
    使用 Linux wget 下载文件到指定目录，并可指定文件名

    :param download_url: 要下载的文件链接
    :param path: 存储文件的目标目录
    :param filename: (可选) 指定保存的文件名，默认为 None，使用 wget 默认文件名
    """
    # 确保目标目录存在
    os.makedirs(path, exist_ok=True)

    # 如果指定了文件名，则设置完整的保存路径
    output_path = os.path.join(path, filename) if filename else None

    # 构造 wget 命令
    command = ["wget", "-c"]  # `-c` 选项支持断点续传
    if output_path:
        command.extend(["-O", output_path])  # 指定文件名
    command.append(download_url)  # 添加下载链接
    command.append("-P") if not filename else None  # 仅在未指定 filename 时使用 `-P`
    command.append(path) if not filename else None  # 仅在未指定 filename 时使用 `-P`

    try:
        # 执行 wget 命令
        subprocess.run([c for c in command if c], check=True)  # 过滤掉 None 值
        print(f"✅ 文件已成功下载到: {output_path if filename else path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")



def download_bank_marketing_dataset():
    # 下载数据集
    path = kagglehub.dataset_download("janiobachmann/bank-marketing-dataset")

    # 目标文件路径（下载后的 CSV 文件）
    file_path = os.path.join(path, 'bank.csv')

    # 确保目标目录存在
    os.makedirs(DATASETS_PATH, exist_ok=True)

    # 目标存储路径
    target_path = os.path.join(DATASETS_PATH, 'bank.csv')

    # 移动文件到 DATASETS_PATH 目录下
    if os.path.exists(file_path):
        shutil.move(file_path, target_path)
        print(f"✅ 文件已成功下载到 {target_path}")
    else:
        print(f"❌ 错误: {file_path} 不存在，下载可能失败。")