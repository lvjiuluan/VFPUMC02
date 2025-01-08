# Hetero-SecureBoost Tutorial

In a hetero-federated learning (vertically partitioned data) setting, multiple parties have different feature sets for the same common user samples. Federated learning enables these parties to collaboratively train a model without sharing their actual data. The model is trained locally at each party, and only model updates are shared, not the actual data. SecureBoost is a specialized tree-boosting framework designed for vertical federated learning. It performs entity alignment under a privacy-preserving protocol and constructs boosting trees across multiple parties using an encryption strategy. It allows for high-quality, lossless model training without needing a trusted third party.

In this tutorial, we will show you how to run a Hetero-SecureBoost task under FATE-2.0 locally without using a FATE-Pipeline. You can refer to this example for local model experimentation, algorithm modification, and testing.

## Setup Hetero-Secureboost Step by Step

To run a Hetero-Secureboost task, several steps are needed:

1. Import required classes in a new python script
2. Prepare tabular data and transform them into fate dataframe
3. Create Launch Function and Run Hetero-Secureboost task

## Import Libs and Write a Python Script

We import these classes for later use.

```python
import pandas as pd
from fate.arch.dataframe import PandasReader
from fate.ml.ensemble.algo.secureboost.hetero.guest import HeteroSecureBoostGuest
from fate.ml.ensemble.algo.secureboost.hetero.host import HeteroSecureBoostHost
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from datetime import datetime
from fate.arch.context import create_context
```

# 