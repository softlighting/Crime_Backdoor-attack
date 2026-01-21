# 数据集格式分析文档

## 1. 数据集概览

本项目使用两个城市犯罪数据集：**NYC (纽约)** 和 **CHI (芝加哥)**。

| 属性 | NYC | CHI |
|------|-----|-----|
| 空间网格 | 16 × 16 | 14 × 12 |
| 区域总数 | 256 | 168 |
| 训练天数 | 608 | 609 |
| 验证天数 | 30 | 30 |
| 测试天数 | 92 | 92 |
| 犯罪类别 | 4 | 4 |
| 稀疏率 | 86.46% | 73.62% |

---

## 2. 数据形状

### 2.1 原始格式 (Pickle文件)

```python
# 数据形状: (row, col, days, offNum)
# row, col: 空间网格维度
# days: 时间天数
# offNum: 犯罪类别数 (4)

# NYC
trn.shape = (16, 16, 608, 4)   # 训练集
val.shape = (16, 16, 30, 4)    # 验证集
tst.shape = (16, 16, 92, 4)    # 测试集

# CHI
trn.shape = (14, 12, 609, 4)   # 训练集
val.shape = (14, 12, 30, 4)    # 验证集
tst.shape = (14, 12, 92, 4)    # 测试集
```

### 2.2 模型输入格式 (重塑后)

```python
# 重塑为: (areaNum, days, offNum)
# areaNum = row * col

# NYC: (256, 608, 4) → (256, days, 4)
# CHI: (168, 609, 4) → (168, days, 4)
```

### 2.3 批次输入格式 (训练时)

```python
# 形状: (batch, areaNum, temporalRange, offNum)
# batch: 批大小 (默认16)
# areaNum: 区域数
# temporalRange: 时间窗口 (默认30天)
# offNum: 犯罪类别数 (4)

# NYC: (B, 256, 30, 4)
# CHI: (B, 168, 30, 4)
```

---

## 3. 犯罪类别映射

```python
offenseMap = {
    'THEFT': 0,           # 盗窃 (类别0)
    'BATTERY': 1,         # 殴打 (类别1)
    'ASSAULT': 2,         # 袭击 (类别2)
    'CRIMINAL DAMAGE': 3  # 刑事毁坏 (类别3)
}
```

---

## 4. 数据统计

### 4.1 NYC 数据集

| 指标 | 值 |
|------|-----|
| 数据类型 | float64 |
| 最小值 | 0.0 |
| 最大值 | 35.0 |
| 均值 | 0.2557 |
| 标准差 | 0.9140 |
| 非零比例 | 13.54% |

**各类别统计:**

| 类别 | 均值 | 最大值 | 非零样本数 |
|------|------|--------|-----------|
| THEFT | 0.1703 | 10 | 17,719 |
| BATTERY | 0.4597 | 35 | 30,221 |
| ASSAULT | 0.1757 | 9 | 17,132 |
| CRIMINAL DAMAGE | 0.2171 | 12 | 19,196 |

### 4.2 CHI 数据集

| 指标 | 值 |
|------|-----|
| 数据类型 | float64 |
| 最小值 | 0.0 |
| 最大值 | 104.0 |
| 均值 | 0.6569 |
| 标准差 | 1.6204 |
| 非零比例 | 26.38% |

**各类别统计:**

| 类别 | 均值 | 最大值 | 非零样本数 |
|------|------|--------|-----------|
| THEFT | 1.0066 | 104 | 32,434 |
| BATTERY | 0.8160 | 17 | 30,141 |
| ASSAULT | 0.3105 | 12 | 19,553 |
| CRIMINAL DAMAGE | 0.4943 | 18 | 25,830 |

---

## 5. 数据归一化

### 5.1 Z-Score 标准化

```python
# 在 DataHandler 中实现
class DataHandler:
    def __init__(self):
        # 计算训练集统计量
        self.mean = np.mean(trnT)
        self.std = np.std(trnT)

    def zScore(self, data):
        """标准化"""
        return (data - self.mean) / self.std

    def zInverse(self, data):
        """反标准化"""
        return data * self.std + self.mean
```

### 5.2 归一化参数

| 数据集 | Mean | Std |
|--------|------|-----|
| NYC | 0.2557 | 0.9140 |
| CHI | 0.6569 | 1.6204 |

---

## 6. 训练/验证/测试划分

### 6.1 NYC 数据集

```
总天数: 730 天 (2年)
├── 训练集: 608 天 (83.3%)
├── 验证集: 30 天 (4.1%)
└── 测试集: 92 天 (12.6%)
```

### 6.2 CHI 数据集

```
总天数: 731 天 (约2年)
├── 训练集: 609 天 (83.3%)
├── 验证集: 30 天 (4.1%)
└── 测试集: 92 天 (12.6%)

时间范围 (从 dataProcessing_CHI.py):
- 2016年1月1日 ~ 2017年8月31日: 训练集 (366+365-92-30=609天)
- 2017年9月: 验证集 (30天)
- 2017年10月-12月: 测试集 (92天)
```

---

## 7. 空间结构

### 7.1 网格化空间

数据将城市划分为规则网格：

```
NYC: 16 × 16 = 256 个区域
     ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
     │0│1│2│3│...                    │
     ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
     │16│17│...                      │
     ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
     │...                            │
     └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

CHI: 14 × 12 = 168 个区域
```

### 7.2 邻接矩阵构建

对于 CSTBA 的空域触发器，需要从网格构建邻接矩阵：

```python
def build_grid_adjacency(row, col, connectivity=8):
    """
    构建网格邻接矩阵

    Args:
        row, col: 网格尺寸
        connectivity: 4 (上下左右) 或 8 (含对角线)

    Returns:
        adj: (areaNum, areaNum) 邻接矩阵
    """
    areaNum = row * col
    adj = np.zeros((areaNum, areaNum))

    for i in range(row):
        for j in range(col):
            idx = i * col + j
            # 4-connectivity
            neighbors = []
            if i > 0: neighbors.append((i-1) * col + j)      # 上
            if i < row-1: neighbors.append((i+1) * col + j)  # 下
            if j > 0: neighbors.append(i * col + (j-1))      # 左
            if j < col-1: neighbors.append(i * col + (j+1))  # 右

            if connectivity == 8:
                # 对角线
                if i > 0 and j > 0: neighbors.append((i-1) * col + (j-1))
                if i > 0 and j < col-1: neighbors.append((i-1) * col + (j+1))
                if i < row-1 and j > 0: neighbors.append((i+1) * col + (j-1))
                if i < row-1 and j < col-1: neighbors.append((i+1) * col + (j+1))

            for n in neighbors:
                adj[idx, n] = 1

    return adj
```

---

## 8. CSTBA 数据格式映射

### 8.1 设计文档中的格式

```
X ∈ R^(T×V×C)
- T: 时间步数
- V: 节点数 (变量数)
- C: 特征维度
```

### 8.2 实际数据映射

```python
# 模型输入: (batch, areaNum, temporalRange, offNum)
# 映射到 CSTBA:
#   B = batch
#   T = temporalRange (30)
#   V = areaNum (256 for NYC, 168 for CHI)
#   C = offNum (4)

# 注意: 维度顺序不同
# CSTBA文档: (B, T, V, C)
# 实际代码: (B, V, T, C)

# 需要转置:
# x_cstba = x.permute(0, 2, 1, 3)  # (B, areaNum, T, C) -> (B, T, areaNum, C)
```

### 8.3 任务类型

| 属性 | 说明 |
|------|------|
| 任务类型 | **回归** (非分类) |
| 标签格式 | 连续值 (犯罪计数) |
| 预测目标 | 下一天各区域各类型犯罪数量 |
| 评估指标 | RMSE, MAE, MAPE |

---

## 9. 数据稀疏性分析

### 9.1 稀疏性分布

由于犯罪是稀疏事件，大部分网格在大部分时间没有犯罪：

```
NYC稀疏率: 86.46%  (约6/7的数据点为0)
CHI稀疏率: 73.62%  (约3/4的数据点为0)
```

### 9.2 对 CSTBA 的影响

1. **时序触发器**: 在稀疏数据上注入触发器更容易被检测（突然出现非零值）
2. **攻击目标**: 应选择原本非零频率较高的区域和时间
3. **隐蔽性**: 触发器幅度应参考数据分布，避免异常值

---

## 10. 数据加载示例

```python
import pickle
import numpy as np

# 加载数据
with open('Datasets/NYC_crime/trn.pkl', 'rb') as f:
    trn = pickle.load(f)

# 基本信息
print(f"Shape: {trn.shape}")  # (16, 16, 608, 4)
print(f"Mean: {trn.mean():.4f}")
print(f"Std: {trn.std():.4f}")

# 重塑为模型输入格式
row, col, days, offNum = trn.shape
areaNum = row * col
trn_reshaped = np.reshape(trn, [areaNum, days, offNum])
print(f"Reshaped: {trn_reshaped.shape}")  # (256, 608, 4)

# 创建时间窗口样本
temporalRange = 30
sample_idx = 50  # 第50天
sample = trn_reshaped[:, sample_idx - temporalRange:sample_idx, :]
print(f"Sample shape: {sample.shape}")  # (256, 30, 4)

# 添加batch维度
batch_sample = np.expand_dims(sample, axis=0)
print(f"Batch sample: {batch_sample.shape}")  # (1, 256, 30, 4)
```

---

## 11. 总结

| 维度 | NYC | CHI | CSTBA映射 |
|------|-----|-----|-----------|
| 空间节点数 V | 256 | 168 | 触发节点选择范围 |
| 时间步数 T | 30 | 30 | 时序触发器长度 |
| 特征维度 C | 4 | 4 | 特征触发维度 |
| 稀疏率 | 86.46% | 73.62% | 影响触发器设计 |
| 任务 | 回归 | 回归 | ASR计算方式 |

**关键要点:**
1. 数据是4D时空张量，需要正确处理维度
2. 任务是回归而非分类，ASR需要定义为预测偏移
3. 高稀疏性需要特别处理触发器设计
4. 空间结构是网格，可构建邻接矩阵用于GNN分析
