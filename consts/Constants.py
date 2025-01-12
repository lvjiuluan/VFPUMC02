import os

# 不同平台跑，配置一下，项目的根路径
# ROOT_PATH = r'D:\PyCharmProjects\VFPUMC02'
# ROOT_PATH = r'/root/semi/VFPUMC02'
ROOT_PATH = r'/root/VFPUMC02'

# 下面的都是基于 ROOT_PATH
DATASETS_PATH = os.path.join(ROOT_PATH, "datasets")
CONFIGS_PATH = os.path.join(ROOT_PATH, "configs")
SCRIPTS_PATH = os.path.join(ROOT_PATH, "scripts")

A_host_path = os.path.join(DATASETS_PATH, 'A_host.csv')
B_guest_path = os.path.join(DATASETS_PATH, 'B_guest.csv')
SBT_CONFIGS_PATH = os.path.join(CONFIGS_PATH, "sbtConfig.yaml")
SBT_PKL_PATH = os.path.join(DATASETS_PATH, 'sbt_result.pkl')
SBT_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'sbt.py')

TOTAL_COUNT = 5000
