import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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