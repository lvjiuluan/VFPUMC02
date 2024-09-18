from enum import Enum

class SplitRatio(Enum):
    RATIO_10_90 = [0.1, 0.9]
    RATIO_20_80 = [0.2, 0.8]
    RATIO_30_70 = [0.3, 0.7]
    RATIO_40_60 = [0.4, 0.6]
    RATIO_50_50 = [0.5, 0.5]
    RATIO_60_40 = [0.6, 0.4]
    RATIO_70_30 = [0.7, 0.3]
    RATIO_80_20 = [0.8, 0.2]
    RATIO_90_10 = [0.9, 0.1]