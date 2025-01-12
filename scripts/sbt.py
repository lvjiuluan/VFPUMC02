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
        bst = HeteroSecureBoostGuest(**config)
    else:
        bst = HeteroSecureBoostHost(**config)

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
    config = load_config(SBT_CONFIGS_PATH)
    result = {}
    if ctx.is_on_guest:
        guest_result = {}
        data = csv_to_df(ctx, B_guest_path)
        bst = train(ctx, data, config)
        model_dict = bst.get_model()
        pred = predict(ctx, data, model_dict)
        pred_df = pred.as_pd_df()
        guest_result['model_dict'] = model_dict
        guest_result['pred_df'] = pred_df
        result['guest_result'] = guest_result
    else:
        host_result = {}
        data = csv_to_df(ctx, A_host_path, has_label=False)
        bst = train(ctx, data, config)
        model_dict = bst.get_model()
        pred = predict(ctx, data, model_dict)
        pred_df = pred.as_pd_df()
        host_result['model_dict'] = model_dict
        host_result['pred_df'] = pred_df
        result['host_result'] = host_result
    save_to_pkl(result, SBT_PKL_PATH)


if __name__ == '__main__':
    launch(run)
