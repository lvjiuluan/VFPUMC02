如果你的数据集 `X` 包含不同类型的列（例如，既有浮点型也有分类型），在使用 CTGAN 时需要对数据进行适当的预处理和配置。CTGAN 可以处理混合类型的数据，但你需要确保数据在输入模型之前是正确格式化的。

以下是一个更复杂的示例，假设 `X` 包含浮点型和分类型数据：

```python
import numpy as np
import pandas as pd
from sdv.tabular import CTGAN
from sklearn.metrics import mean_squared_error

# 假设你有一个二维数据 X，包含浮点型和分类型数据
n, m = 100, 10
np.random.seed(0)

# 生成随机浮点型数据
float_data = np.random.rand(n, m // 2)

# 生成随机分类型数据（假设有3个类别）
category_data = np.random.randint(0, 3, size=(n, m // 2))

# 合并为一个 DataFrame
X = pd.DataFrame(np.hstack((float_data, category_data)), columns=[f'col_{i}' for i in range(m)])

# 将分类数据转换为字符串（或类别类型）
for i in range(m // 2, m):
    X[f'col_{i}'] = X[f'col_{i}'].astype(str)

# 水平切分数据
X_half_train = X.iloc[:n//2, :]
X_half_test = X.iloc[n//2:, :]

# 初始化并训练 CTGAN
ctgan = CTGAN()
ctgan.fit(X_half_train)

# 生成合成数据
X_half_gen = ctgan.sample(n//2)

# 构造完整数据
X_constr = pd.concat([X_half_train, X_half_gen], ignore_index=True)

# 评估生成效果
# 由于数据包含分类和浮点型，评估可以分开进行
# 这里仅对浮点型数据计算 MSE
mse = mean_squared_error(X.iloc[:, :m//2], X_constr.iloc[:, :m//2])
print(f'Mean Squared Error for float columns: {mse}')

# 你可以根据需要使用其他指标进行评估
```

### 关键点

1. **数据预处理**：确保分类数据在输入 CTGAN 之前被正确标记为分类数据（例如，转换为字符串或类别类型）。

2. **模型训练**：CTGAN 可以处理混合类型的数据，但在训练时需要确保数据的格式正确。

3. **评估**：对于混合类型的数据，评估生成效果时可以分别对浮点型和分类型数据进行评估。对于浮点型数据，可以使用均方误差（MSE）等指标；对于分类型数据，可以使用分类准确率、分布相似性等指标。

根据你的具体需求和数据特性，可以调整评估方法和指标。