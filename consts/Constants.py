import os

# 不同平台跑，配置一下，项目的根路径
# ROOT_PATH = r'D:\PyCharmProjects\VFPUMC02'
# ROOT_PATH = r'/root/semi/VFPUMC02'
ROOT_PATH = r'/root/VFPUMC02'
# ROOT_PATH = r'C:\Users\Administrator\PycharmProjects\VFPUMC02'

# 下面的都是基于 ROOT_PATH
DATASETS_PATH = os.path.join(ROOT_PATH, "datasets")
CONFIGS_PATH = os.path.join(ROOT_PATH, "configs")
SCRIPTS_PATH = os.path.join(ROOT_PATH, "scripts")

A_host_train_path = os.path.join(DATASETS_PATH, 'A_host_train.csv')
B_guest_train_path = os.path.join(DATASETS_PATH, 'B_guest_train.csv')
A_host_test_path = os.path.join(DATASETS_PATH, 'A_host_test.csv')
B_guest_test_path = os.path.join(DATASETS_PATH, 'B_guest_test.csv')

SBT_CONFIGS_PATH = os.path.join(CONFIGS_PATH, "sbtConfig.yaml")
SBT_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'sbt.py')
SBT_PKL_GUEST_PATH = os.path.join(DATASETS_PATH, 'sbt_guest_result.pkl')
SBT_PKL_HOST_PATH = os.path.join(DATASETS_PATH, 'sbt_host_result.pkl')
TOTAL_COUNT = 5000
