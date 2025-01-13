# 添加项目的根路径，不同平台运行不一样
import sys
sys.path.append('D:\PyCharmProjects\VFPUMC02')
sys.path.append(r'C:\Users\Administrator\PycharmProjects\VFPUMC02')
sys.path.append(r'/root/VFPUMC02')

import os

import pandas as pd
from fate.arch.dataframe import PandasReader
from fate.ml.ensemble.algo.secureboost.hetero.guest import HeteroSecureBoostGuest
from fate.ml.ensemble.algo.secureboost.hetero.host import HeteroSecureBoostHost
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from datetime import datetime
from fate.arch.context import create_context
from fate.arch.launchers.multiprocess_launcher import launch
import yaml
from utils.FateUtils import *
from utils.pklUtils import *

from consts.Constants import *




def train(ctx: Context, data: DataFrame, config):
    if ctx.is_on_guest:
        # 过滤掉与训练无关的参数
        filtered_config = filter_params_for_class(HeteroSecureBoostGuest, config)
        bst = HeteroSecureBoostGuest(**filtered_config)
    else:
        # 过滤掉与训练无关的参数
        filtered_config = filter_params_for_class(HeteroSecureBoostHost, config)
        bst = HeteroSecureBoostHost(**filtered_config)

    bst.fit(ctx, data)

    return bst


def predict(ctx: Context, data: DataFrame, model_dict: dict):
    if ctx.is_on_guest:
        bst = HeteroSecureBoostGuest()
    else:
        bst = HeteroSecureBoostHost()
    bst.from_model(model_dict)
    return bst.predict(ctx, data)


def csv_to_df(ctx, file_path, has_label=True):
    df = pd.read_csv(file_path)
    if has_label:
        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", label_name="y", dtype="float32")
    else:
        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", dtype="float32")

    fate_df = reader.to_frame(ctx, df)
    return fate_df


def run(ctx):
    # 加载配置
    config = load_config(SBT_CONFIGS_PATH)
    
    # 初始化结果字典
    result = {}
    
    # 判断是否为 guest
    is_guest = ctx.is_on_guest
    
    # 根据角色加载数据
    data_path = B_guest_path if is_guest else A_host_path
    data = csv_to_df(ctx, data_path, has_label=is_guest)
    
    # 训练模型
    bst = train(ctx, data, config)
    model_dict = bst.get_model()
    
    # 预测
    pred = predict(ctx, data, model_dict)
    if pred is not None:
        pred_df = pred.as_pd_df()
    else:
        pred_df = None
    
    # 构建结果
    role_key = 'guest' if is_guest else 'host'
    result[role_key] = {
        'model_dict': model_dict,
        'pred_df': pred_df,
        'ctx': ctx
    }

    # 设置保存路径
    SBT_PKL_PATH = SBT_PKL_GUEST_PATH if is_guest else SBT_PKL_HOST_PATH
    
    # 保存结果
    add_to_dict_pkl(result, SBT_PKL_PATH)

def run(ctx):
    if ctx.is_on_guest:
        config = load_config(SBT_CONFIGS_PATH)
        result = {}
        train_data = csv_to_df(ctx, B_guest_train_path)
        test_data = csv_to_df(ctx, B_guest_test_path,has_label=False)
        bst = train(ctx, train_data, config)
        model_dict = bst.get_model()
        pred = predict(ctx, test_data, model_dict)
        if pred is not None:
            pred_df = pred.as_pd_df()
        else:
            pred_df = None
        result['model_dict'] = model_dict
        result['pred_df'] = pred_df
        save_to_pkl(result,SBT_PKL_GUEST_PATH)
    else:
        config = load_config(SBT_CONFIGS_PATH)
        result = {}
        train_data = csv_to_df(ctx, A_host_train_path, has_label=False)
        test_data = csv_to_df(ctx, A_host_train_path, has_label=False)
        bst = train(ctx, train_data, config)
        model_dict = bst.get_model()
        pred = predict(ctx, test_data, model_dict)
        if pred is not None:
            pred_df = pred.as_pd_df()
        else:
            pred_df = None
        result['model_dict'] = model_dict
        result['pred_df'] = pred_df
        save_to_pkl(result,SBT_PKL_HOST_PATH)
        


if __name__ == '__main__':
    launch(run)
    
