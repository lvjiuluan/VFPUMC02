在使用`scikit-learn`进行逻辑回归和线性回归时，L2正则化是通过一个称为`C`（逻辑回归）或`alpha`（线性回归）的超参数来控制的。下面是如何使用Python代码来演示这两个模型，并解释每个模型的超参数。

### 逻辑回归（Logistic Regression）

逻辑回归用于分类任务。`scikit-learn`中的`LogisticRegression`类提供了L2正则化。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型
model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=100)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

#### 超参数解释：
- `penalty`: 正则化类型，'l2'表示使用L2正则化。
- `C`: 正则化强度的倒数，较小的值指定更强的正则化。默认值是1.0。
- `solver`: 优化算法，'lbfgs'适用于小型数据集的多类分类。
- `max_iter`: 最大迭代次数，默认是100。

### 线性回归（Ridge Regression）

线性回归用于回归任务。`scikit-learn`中的`Ridge`类提供了L2正则化。

```python
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据集
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建岭回归模型
model = Ridge(alpha=1.0, solver='auto', max_iter=None)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

#### 超参数解释：
- `alpha`: 正则化强度，较大的值指定更强的正则化。默认值是1.0。
- `solver`: 求解器，'auto'表示自动选择合适的求解器。
- `max_iter`: 最大迭代次数，默认是None，表示不限制迭代次数。

这些示例展示了如何使用`scikit-learn`中的逻辑回归和岭回归模型，并解释了每个模型的主要超参数。根据数据集的不同，可能需要调整这些超参数以获得最佳性能。


当然，下面是包含默认值的逻辑回归和岭回归模型超参数的总结表格：

### 逻辑回归 (Logistic Regression)

| 超参数          | 默认值       | 可能取值                           | 作用说明                                                                 |
|----------------|------------|--------------------------------|----------------------------------------------------------------------|
| `penalty`      | `'l2'`     | `'l1'`, `'l2'`, `'elasticnet'`, `'none'` | 正则化类型，'l2'表示L2正则化，'l1'表示L1正则化，'elasticnet'是L1和L2的组合，'none'表示无正则化。 |
| `C`            | `1.0`      | 任意正浮点数                        | 正则化强度的倒数，较小的值指定更强的正则化。                                           |
| `solver`       | `'lbfgs'`  | `'newton-cg'`, `'lbfgs'`, `'liblinear'`, `'sag'`, `'saga'` | 优化算法选择，不同的solver适用于不同的数据规模和正则化类型。                           |
| `max_iter`     | `100`      | 正整数                            | 最大迭代次数，控制优化过程的迭代上限。                                               |
| `tol`          | `1e-4`     | 任意正浮点数                        | 优化过程的容差，决定收敛的精度。                                                   |
| `class_weight` | `None`     | `None`, `'balanced'`, 字典或'balanced' | 类别权重，用于处理类别不平衡问题。'balanced'会根据数据自动调整权重。                  |

### 岭回归 (Ridge Regression)

| 超参数          | 默认值       | 可能取值                           | 作用说明                                                                 |
|----------------|------------|--------------------------------|----------------------------------------------------------------------|
| `alpha`        | `1.0`      | 任意正浮点数                        | 正则化强度，较大的值指定更强的正则化。                                           |
| `solver`       | `'auto'`   | `'auto'`, `'svd'`, `'cholesky'`, `'lsqr'`, `'sparse_cg'`, `'sag'`, `'saga'` | 求解器选择，不同的solver适用于不同的数据规模和特性。                             |
| `max_iter`     | `None`     | 正整数或`None`                     | 最大迭代次数，适用于迭代求解器。`None`表示不限制迭代次数。                            |
| `tol`          | `1e-3`     | 任意正浮点数                        | 优化过程的容差，决定收敛的精度。                                                   |
| `random_state` | `None`     | 整数、`RandomState`实例或`None`     | 用于在某些solver中设置随机数生成器的种子，确保结果的可重复性。                          |

这些默认值是`scikit-learn`中常用的设置，通常可以作为模型调优的起点。根据具体的数据集和问题需求，可能需要调整这些超参数以优化模型的性能。