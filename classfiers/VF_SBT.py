from classfiers.VF_BASE import VF_BASE_CLF
from utils.FateUtils import *
from consts.Constants import *
from utils.pklUtils import *
import subprocess


class VF_SBT(VF_BASE_CLF):
    def __int__(self, config):
        sbtConfig = load_config(SBT_CONFIGS_PATH)
        sbtConfig.update(config)
        self.config = sbtConfig
        self.result = None
        save_config(sbtConfig, SBT_CONFIGS_PATH)

    def fit(self, XA, XB, y):
        A_df, B_df = fate_construct_df(XA, XB, y)
        save_host_guest_dataframes(A_df, B_df, A_host_path, B_guest_path)
        SBT_SCRIPT_PATH = '/root/script/sbt.py'
        # 构建命令
        command = [
            'python',
            SBT_SCRIPT_PATH,
            '--parties',
            'guest:9999',
            'host:10000',
            '--log_level',
            self.config['log_level']
        ]
        #  运行脚本，但不打印脚本输出日志。脚本完成后才打印 “SBT 训练完成”
        try:
            # 使用 stdout 和 stderr 重定向到 DEVNULL，避免在运行过程中打印任何日志
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # 打印脚本输出
            print("sbt.py 输出:")
            print(result.stdout)
            # 脚本执行完成后才打印
            print("SBT 训练完成")
        except subprocess.CalledProcessError as cpe:
            # 如果脚本执行返回非 0 退出码，会抛出 CalledProcessError
            print(f"sbt.py 运行失败，退出状态码: {cpe.returncode}")
            # 也可以视需求打印错误输出或采取其他处理方式
            # 例如: print(f"错误信息:\n{cpe.stderr}")
            raise
        except FileNotFoundError:
            print(f"找不到脚本文件: {SBT_SCRIPT_PATH}")
            raise
        except Exception as e:
            print(f"运行 sbt.py 时发生未知错误: {e}")
            raise
        result = load_from_pkl(SBT_PKL_PATH)
        self.result = result
        print("VF_SBT训练结束")

    def predict(self, XA, XB):
        pass

    def predict_proba(self, XA, XB):
        pass
