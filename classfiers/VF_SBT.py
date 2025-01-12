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
        save_host_guest_dataframes(A_df, B_df, A_host_path, B_guest_path)
        # 构建命令
        command = [
            sys.executable,
            SBT_SCRIPT_PATH,
            '--parties',
            'guest:9999',
            'host:10000',
            '--log_level',
            self.config['log_level']
        ]
        
        # 使用subprocess.run来执行命令
        result = subprocess.run(command, capture_output=True, text=True)
        
        # 打印输出结果
        print("Standard Output:", result.stdout)
        print("Standard Error:", result.stderr)
        
        # 检查命令是否成功运行
        if result.returncode == 0:
            print("Command executed successfully.")
        else:
            print("Command failed with return code:", result.returncode)
        result = load_from_pkl(SBT_PKL_PATH)
        self.result = result
        print("VF_SBT训练结束")

    def predict(self, XA, XB):
        pass

    def predict_proba(self, XA, XB):
        pass
