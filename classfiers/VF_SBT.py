from classfiers.VF_BASE import VF_BASE_CLF
from utils.FateUtils import *
from consts.Constants import *
from utils.pklUtils import *
import subprocess
import sys


class VF_SBT(VF_BASE_CLF):
    def __init__(self, config):
        sbtConfig = load_config(SBT_CONFIGS_PATH)
        sbtConfig.update(config)
        self.config = sbtConfig
        self.result = None
        save_config(sbtConfig, SBT_CONFIGS_PATH)

    def fit(self, XA, XB, y):
        A_df, B_df = fate_construct_df(XA, XB, y)
        save_host_guest_dataframes(A_df, B_df, A_host_train_path, B_guest_train_path)
        print("VF_SBT训练结束")

    def predict(self, XA, XB):
        pass

    def predict_proba(self, XA, XB):
        A_df, B_df = fate_construct_df(XA, XB)
        save_host_guest_dataframes(A_df, B_df, A_host_test_path, B_guest_test_path)
        execute_sbt_command(self.config)