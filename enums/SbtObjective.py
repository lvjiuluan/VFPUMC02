from enum import Enum


class SbtObjective(Enum):
    BINARY_BCE = 'binary:bce'
    MULTI_CE = "multi:ce"
    REGRESSION_L2 = "regression:l2"


