# 项目结构分析报告

## 1. 项目概述

**项目名称**: STHSL (Spatial-Temporal Hypergraph Self-Supervised Learning)

**来源**: ICDE 2022 论文 - "Spatial-Temporal Hypergraph Self-Supervised Learning for Crime Prediction"

**功能**: 使用超图自监督学习方法进行城市犯罪预测

**目标模型架构**: 时空超图神经网络 (ST-Hypergraph NN)

---

## 2. 目录结构

```
Crime_Backdoor-attack/
├── Datasets/                          # 数据集目录
│   ├── NYC_crime/                     # 纽约犯罪数据集
│   │   ├── trn.pkl                    # 训练集
│   │   ├── val.pkl                    # 验证集
│   │   ├── tst.pkl                    # 测试集
│   │   └── NYC_Crime.zip              # 原始数据压缩包
│   └── CHI_crime/                     # 芝加哥犯罪数据集
│       ├── trn.pkl                    # 训练集
│       ├── val.pkl                    # 验证集
│       ├── tst.pkl                    # 测试集
│       ├── CHI_Crime.zip              # 原始数据压缩包
│       └── dataProcessing_CHI.py      # 数据预处理脚本
├── Save/                              # 模型保存目录
│   ├── NYC/                           # NYC模型检查点
│   └── CHI/                           # CHI模型检查点
├── docs/                              # 文档目录 (新建)
├── model.py                           # 模型定义 (STHSL)
├── train.py                           # 训练脚本
├── test.py                            # 测试脚本
├── engine.py                          # 训练引擎/优化器
├── DataHandler.py                     # 数据加载器
├── Params.py                          # 参数配置
├── utils.py                           # 工具函数
├── README.md                          # 项目说明
├── CSTBA_Algorithm_Design.docx        # CSTBA算法设计文档
├── CSTBA_Implementation_Roadmap.md    # CSTBA实现路线图
└── CSTBA_Architecture.jsx             # CSTBA架构图(React组件)
```

---

## 3. 关键组件详解

### 3.1 数据加载模块 - `DataHandler.py`

**位置**: `/DataHandler.py`

**功能**:
- 加载 pickle 格式的时空犯罪数据
- 支持 NYC 和 CHI 两个数据集
- 提供 z-score 标准化和反标准化
- 计算数据稀疏性掩码

**关键属性**:
```python
- trnT, valT, tstT: 训练/验证/测试数据张量
- mean, std: 数据统计量 (用于标准化)
- mask1-4: 稀疏性分组掩码
- args.row, args.col: 空间网格尺寸
- args.areaNum: 区域总数 (row * col)
- args.trnDays, valDays, tstDays: 各集天数
```

### 3.2 模型定义 - `model.py`

**位置**: `/model.py`

**主模型**: `STHSL` 类

**架构组件**:

| 组件 | 类名 | 功能 |
|------|------|------|
| 局部空间卷积 | `spa_cnn_local` | 3D CNN提取局部空间特征 |
| 局部时间卷积 | `tem_cnn_local` | 3D CNN提取局部时间特征 |
| 全局超图 | `Hypergraph` | 超图神经网络建模全局依赖 |
| 超图互信息 | `Hypergraph_Infomax` | 自监督学习模块 |
| 全局时间卷积 | `tem_cnn_global` | 全局时间特征提取 |
| 特征变换 | `Transform_3d` | BatchNorm + Conv变换 |

**模型流程**:
```
输入 (B, row*col, T, 4)
    → dimConv_in 升维
    → 局部分支: spa_cnn_local × 2 → tem_cnn_local × 2 → out_local
    → 全局分支: Hypergraph_Infomax → tem_cnn_global × 4 → out_global
输出: out_local, out_global, 中间表示
```

### 3.3 训练脚本 - `train.py`

**位置**: `/train.py`

**功能**:
- 初始化训练器 (engine.trainer)
- 执行训练循环
- 定期评估并保存最优模型

**关键超参数** (来自 Params.py):
- epochs: 25
- batch_size: 16
- learning_rate: 1e-3
- temporalRange: 30 (时间窗口)

### 3.4 训练引擎 - `engine.py`

**位置**: `/engine.py`

**trainer 类功能**:
- `sampleTrainBatch()`: 采样训练批次
- `sampTestBatch()`: 采样测试批次
- `train()`: 单轮训练，计算复合损失
- `eval()`: 评估模型性能

**损失函数组成**:
```python
loss = Informax_loss * ir + infoNCEloss * cr + MSE_loss(local) + MSE_loss(global)
```

### 3.5 配置文件 - `Params.py`

**位置**: `/Params.py`

**关键参数**:
```python
# 训练参数
lr = 1e-3              # 学习率
weight_decay = 1e-4    # 权重衰减
batch = 16             # 批大小
epoch = 25             # 训练轮数

# 模型参数
latdim = 16            # 隐藏维度
temporalRange = 30     # 时间窗口长度
cateNum = 4            # 犯罪类别数
hyperNum = 128         # 超边数量
kernelSize = 3         # 卷积核大小

# 自监督参数
cr = 0.8               # 对比学习损失权重
ir = 1                 # 互信息损失权重
t = 0.05               # 温度参数

# Dropout
dropRateL = 0.2        # 局部编码器dropout
dropRateG = 0.1        # 全局编码器dropout
```

### 3.6 工具函数 - `utils.py`

**位置**: `/utils.py`

**功能函数**:
| 函数 | 功能 |
|------|------|
| `cal_loss_r()` | 计算回归MSE损失 |
| `cal_metrics_r()` | 计算RMSE/MAE/MAPE指标 |
| `cal_metrics_r_mask()` | 带掩码的指标计算 |
| `Informax_loss()` | 互信息最大化损失 |
| `infoNCEloss()` | 对比学习NCE损失 |
| `seed_torch()` | 设置随机种子 |
| `makePrint()` | 格式化打印结果 |

### 3.7 数据集位置

**NYC 犯罪数据集**: `/Datasets/NYC_crime/`
- 训练集: `trn.pkl`
- 验证集: `val.pkl`
- 测试集: `tst.pkl`

**CHI 犯罪数据集**: `/Datasets/CHI_crime/`
- 训练集: `trn.pkl`
- 验证集: `val.pkl`
- 测试集: `tst.pkl`
- 预处理脚本: `dataProcessing_CHI.py`

---

## 4. 数据格式概述

### 4.1 原始数据形状

```python
# 从 pickle 加载后的形状
trnT.shape = (row, col, trnDays, offNum)
valT.shape = (row, col, valDays, offNum)
tstT.shape = (row, col, tstDays, offNum)

# 其中:
# - row, col: 空间网格尺寸 (由数据自动推断)
# - trnDays, valDays, tstDays: 各数据集的天数
# - offNum = 4: 犯罪类别数量
```

### 4.2 犯罪类别映射

```python
offenseMap = {
    'THEFT': 0,           # 盗窃
    'BATTERY': 1,         # 殴打
    'ASSAULT': 2,         # 袭击
    'CRIMINAL DAMAGE': 3  # 刑事毁坏
}
```

### 4.3 数据归一化方式

```python
# Z-score 标准化
normalized = (data - mean) / std

# 反标准化
original = data * std + mean
```

---

## 5. 运行方式

### 训练模型

```bash
# NYC数据集
python train.py --data NYC

# CHI数据集
python train.py --data CHI
```

### 测试模型

```bash
# NYC数据集
python test.py --data NYC --checkpoint ./Save/NYC/model.pth

# CHI数据集
python test.py --data CHI --checkpoint ./Save/CHI/model.pth
```

---

## 6. CSTBA集成要点

### 6.1 可复用组件

1. **数据加载**: `DataHandler` 类可直接复用
2. **模型架构**: `STHSL` 作为目标攻击模型
3. **评估指标**: `utils.py` 中的指标计算函数
4. **训练循环**: `engine.py` 中的训练逻辑

### 6.2 需要扩展的部分

1. **图结构**: 当前模型使用超图而非标准图，需要：
   - 从空间网格构建邻接矩阵
   - 或者利用超图的邻接张量 `Hypergraph.adj`

2. **触发器注入点**:
   - 时序触发: 注入到输入张量 `(B, areaNum, T, 4)`
   - 空域触发: 修改超图邻接矩阵或节点特征

3. **模型适配**:
   - STHSL 使用超图而非标准 GNN
   - 需要先写出简单的STGCN网络，并根据stgcn网络等标准的时空模式来进行触发器注入

### 6.3 关键发现

| 项目 | 发现 |
|------|------|
| 模型类型 | 超图神经网络 (非标准GNN) |
| 任务类型 | 回归任务 (犯罪数量预测) |
| 空间结构 | 网格化 (row × col) |
| 时间结构 | 滑动窗口 (temporalRange=30) |
| 自监督 | 使用 Infomax + InfoNCE |

---

## 7. 总结

本代码库实现了 STHSL 时空犯罪预测模型，具有以下特点：

1. **双分支架构**: 局部CNN编码器 + 全局超图编码器
2. **自监督学习**: Infomax 和 InfoNCE 辅助任务
3. **网格化空间**: 将城市划分为 row × col 网格
4. **多类别预测**: 同时预测4种犯罪类型

对于 CSTBA 算法实现，需要：
- 扩展数据加载器支持触发器注入
- 构建/提取邻接矩阵用于空域分析
- 适配触发器模块以匹配超图结构

