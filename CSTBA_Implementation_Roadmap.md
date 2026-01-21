# CSTBA算法实现计划表
## Collaborative Spatio-Temporal Backdoor Trigger Algorithm Implementation Roadmap

**目标**：根据《协同式时空后门触发器注入算法技术设计文档》，在现有GitHub代码库基础上完整实现CSTBA算法。

**执行者**：Claude Code
**预计工作量**：分6个Phase，每个Phase包含多个Task

---

## Phase 0: 环境准备与代码库理解

### Task 0.1: 代码库结构分析
**指令**：扫描整个代码库，理解现有项目结构。执行以下操作：
1. 列出项目根目录下所有文件和文件夹结构（至少3层深度）
2. 识别并记录以下关键组件的位置：
   - 数据加载模块（data loader）
   - 模型定义文件（ST-GCN, DCRNN, Graph WaveNet, ASTGCN等）
   - 训练脚本（train.py或类似文件）
   - 配置文件（config.yaml或类似文件）
   - 数据集存放位置
3. 阅读README.md了解项目的运行方式
4. 输出一份项目结构报告，标注每个关键文件的功能

**输出物**：`docs/project_structure_analysis.md`

### Task 0.2: 依赖环境检查
**指令**：检查并确保所有必要的依赖已安装：
1. 检查requirements.txt或environment.yaml
2. 确认以下库的可用性：PyTorch, PyTorch Geometric (或DGL), NumPy, SciPy, NetworkX, scikit-learn
3. 如缺少必要依赖，更新requirements.txt
4. 验证CUDA可用性（如适用）

**输出物**：更新后的`requirements.txt`，环境验证日志

### Task 0.3: 数据集格式分析
**指令**：分析现有数据集的格式和结构：
1. 找到数据集文件（.npz, .h5, .csv等）
2. 加载数据集并输出以下信息：
   - 时间序列形状：(样本数, 时间步T, 节点数V, 特征维度C)
   - 邻接矩阵形状和稀疏性
   - 标签格式（回归值 or 分类标签）
   - 训练/验证/测试集划分比例
3. 记录数据归一化方式
4. 创建数据格式文档

**输出物**：`docs/dataset_format.md`

---

## Phase 1: 核心数据结构与工具函数

### Task 1.1: 创建CSTBA模块目录结构
**指令**：在项目中创建CSTBA算法的目录结构：
```
cstba/
├── __init__.py
├── config/
│   └── cstba_config.yaml          # 攻击参数配置
├── modules/
│   ├── __init__.py
│   ├── temporal_trigger.py        # TTM模块
│   ├── spatial_trigger.py         # STM模块
│   ├── synergy_optimizer.py       # 协同优化器
│   └── trigger_injection.py       # 触发注入引擎
├── utils/
│   ├── __init__.py
│   ├── node_selection.py          # 关键节点识别
│   ├── frequency_utils.py         # 频域分析工具
│   └── metrics.py                 # ASR、BA Drop等指标计算
├── attacks/
│   ├── __init__.py
│   └── cstba_attack.py            # 主攻击类
└── scripts/
    ├── train_backdoor.py          # 后门模型训练脚本
    └── evaluate_attack.py         # 攻击效果评估脚本
```

**输出物**：完整的目录结构和空的`__init__.py`文件

### Task 1.2: 实现配置管理
**指令**：创建`cstba/config/cstba_config.yaml`，包含以下参数：
```yaml
# 攻击基础参数
attack:
  poison_rate: 0.1                 # 投毒比例 (5%-15%)
  target_label: "high_risk"        # 攻击目标标签
  trigger_mode: "joint"            # temporal_only / spatial_only / joint

# 时序触发器参数 (TTM)
temporal:
  trigger_type: "periodic"         # periodic / spike / distributed
  injection_window_ratio: 0.33     # 在时间窗口后1/3注入
  amplitude_ratio: 0.1             # 触发幅度为数据标准差的比例
  frequency_constraint: true       # 是否启用频域约束
  smoothness_weight: 0.1           # 时间平滑损失权重

# 空域触发器参数 (STM)
spatial:
  trigger_type: "feature"          # feature / topology / both
  num_trigger_nodes: 5             # 触发节点数量
  node_selection: "centrality"     # centrality / random / target_proximity
  homophily_constraint: true       # 是否启用同质性伪装
  centrality_weights:              # 中心性融合权重
    pagerank: 0.4
    betweenness: 0.3
    eigenvector: 0.3

# 协同优化参数
synergy:
  lambda_attack: 1.0               # 攻击损失权重
  lambda_stealth: 0.5              # 隐蔽性损失权重
  lambda_synergy: 0.3              # 协同损失权重
  mi_estimation: "mine"            # 互信息估计方法

# 训练参数
training:
  surrogate_epochs: 50             # 代理模型预训练轮数
  bilevel_iterations: 100          # 双层优化迭代次数
  inner_lr: 0.001                  # 下层优化学习率
  outer_lr: 0.0001                 # 上层优化学习率
```

同时创建配置加载函数`cstba/config/config_loader.py`。

**输出物**：`cstba_config.yaml` 和 `config_loader.py`

### Task 1.3: 实现评估指标函数
**指令**：在`cstba/utils/metrics.py`中实现以下指标计算函数：

```python
def compute_asr(model, poisoned_loader, target_label, device):
    """计算攻击成功率 (Attack Success Rate)
    ASR = 含触发器样本中预测为目标标签的比例
    """
    pass

def compute_ba_drop(model, clean_loader, baseline_accuracy, device):
    """计算良性准确率下降 (Benign Accuracy Drop)
    BA Drop = baseline_accuracy - current_accuracy
    """
    pass

def compute_stealthiness_score(original_data, poisoned_data):
    """计算隐蔽性得分
    包含：时间平滑度、频谱相似度、特征分布KL散度
    """
    pass

def evaluate_attack_effectiveness(model, clean_loader, poisoned_loader, 
                                   target_label, baseline_metrics, device):
    """综合评估攻击效果
    返回：ASR, BA Drop, Stealthiness Score
    """
    pass
```

确保函数兼容回归任务（MAE/RMSE偏移作为ASR）和分类任务。

**输出物**：完整的`metrics.py`

---

## Phase 2: 时序触发器模块 (TTM) 实现

### Task 2.1: 实现基础时序触发器生成器
**指令**：在`cstba/modules/temporal_trigger.py`中实现`TemporalTriggerModule`类：

```python
class TemporalTriggerModule(nn.Module):
    """时序触发器模块
    
    核心功能：
    1. 基于GAT的触发器生成器，建模多变量耦合关系
    2. 支持三种触发模式：周期微振荡、突变-恢复、分布式片段
    3. 形状感知归一化损失确保频域隐形
    """
    
    def __init__(self, input_dim, hidden_dim, num_variables, 
                 trigger_type='periodic', injection_ratio=0.33):
        """
        Args:
            input_dim: 输入特征维度C
            hidden_dim: 隐藏层维度
            num_variables: 变量数V（节点数）
            trigger_type: 触发器类型
            injection_ratio: 在时间窗口的哪个比例开始注入
        """
        pass
    
    def generate_trigger(self, x):
        """生成时序触发扰动
        Args:
            x: 原始时间序列 [B, T, V, C]
        Returns:
            delta_t: 时序触发扰动 [B, T, V, C]
        """
        pass
    
    def compute_smoothness_loss(self, delta_t):
        """计算时间平滑损失
        L_smooth = sum_t ||delta_t - delta_{t-1}||^2
        """
        pass
    
    def compute_frequency_loss(self, original, perturbed):
        """计算频域一致性损失
        通过FFT比较高频分量能量分布
        """
        pass
```

**关键实现要点**：
1. GAT生成器需要接收节点特征和图结构，输出与输入同形状的扰动
2. 周期触发器：使用可学习频率和振幅的正弦波叠加
3. 突变-恢复触发器：短序列内的尖峰模式，通过高斯函数参数化
4. 分布式片段触发器：通过可学习的mask决定哪些时间步注入

**输出物**：完整的`temporal_trigger.py`

### Task 2.2: 实现频域工具函数
**指令**：在`cstba/utils/frequency_utils.py`中实现频域分析工具：

```python
def compute_fft_spectrum(signal):
    """计算信号的傅里叶频谱"""
    pass

def compute_high_frequency_energy(spectrum, threshold_ratio=0.7):
    """计算高频分量能量占比"""
    pass

def shape_aware_normalization(trigger, original_signal):
    """形状感知归一化
    调整触发器使其频率分布与原始信号一致
    """
    pass

def spectral_similarity(signal1, signal2):
    """计算两个信号的频谱相似度（用于隐蔽性评估）"""
    pass
```

**输出物**：完整的`frequency_utils.py`

### Task 2.3: 单元测试TTM模块
**指令**：创建`tests/test_temporal_trigger.py`，验证TTM模块功能：
1. 测试触发器生成的形状正确性
2. 测试三种触发模式的输出差异
3. 测试平滑损失和频域损失的计算
4. 测试触发器幅度是否在约束范围内
5. 可视化生成的触发器波形（保存为图片）

**输出物**：`tests/test_temporal_trigger.py`，测试通过日志

---

## Phase 3: 空域触发器模块 (STM) 实现

### Task 3.1: 实现关键节点识别算法
**指令**：在`cstba/utils/node_selection.py`中实现节点选择策略：

```python
class NodeSelector:
    """关键节点选择器
    
    使用多指标融合策略识别图中的关键节点
    """
    
    def __init__(self, adj_matrix, centrality_weights=None):
        """
        Args:
            adj_matrix: 邻接矩阵 [V, V]
            centrality_weights: {'pagerank': 0.4, 'betweenness': 0.3, 'eigenvector': 0.3}
        """
        pass
    
    def compute_pagerank(self):
        """计算PageRank中心性"""
        pass
    
    def compute_betweenness(self):
        """计算介数中心性"""
        pass
    
    def compute_eigenvector_centrality(self):
        """计算特征向量中心性"""
        pass
    
    def compute_composite_score(self):
        """计算复合中心性得分
        S(v) = α·PageRank(v) + β·Betweenness(v) + γ·Eigenvector(v)
        """
        pass
    
    def compute_message_passing_influence(self, k_hops=3):
        """计算k跳消息传递影响范围"""
        pass
    
    def select_trigger_nodes(self, num_nodes, target_nodes=None):
        """选择触发节点
        Args:
            num_nodes: 需要选择的节点数
            target_nodes: 目标攻击区域节点（可选，用于计算桥接节点）
        Returns:
            selected_indices: 选中的节点索引列表
        """
        pass
```

**关键实现要点**：
1. 使用NetworkX计算各种中心性指标
2. 如果提供了target_nodes，优先选择能有效到达目标的桥接节点
3. 避免选择孤立节点或度数过低的节点

**输出物**：完整的`node_selection.py`

### Task 3.2: 实现空域触发器生成模块
**指令**：在`cstba/modules/spatial_trigger.py`中实现`SpatialTriggerModule`类：

```python
class SpatialTriggerModule(nn.Module):
    """空域触发器模块
    
    核心功能：
    1. 特征触发器：修改关键节点的特征向量
    2. 拓扑触发器：注入虚假边或修改边权重
    3. 同质性伪装：确保修改后特征符合邻居分布
    """
    
    def __init__(self, feature_dim, num_nodes, trigger_type='feature',
                 homophily_constraint=True):
        """
        Args:
            feature_dim: 节点特征维度
            num_nodes: 图中节点总数
            trigger_type: 'feature' / 'topology' / 'both'
            homophily_constraint: 是否启用同质性约束
        """
        pass
    
    def set_trigger_nodes(self, trigger_node_indices):
        """设置触发节点索引"""
        pass
    
    def generate_feature_trigger(self, node_features, adj_matrix):
        """生成特征触发器
        Args:
            node_features: 节点特征 [V, C] 或 [B, V, C]
            adj_matrix: 邻接矩阵 [V, V]
        Returns:
            delta_f: 特征扰动 [V, C]，仅在trigger_nodes上非零
        """
        pass
    
    def generate_topology_trigger(self, adj_matrix, target_nodes):
        """生成拓扑触发器
        在trigger_nodes和target_nodes之间注入边
        Returns:
            delta_A: 邻接矩阵扰动 [V, V]
        """
        pass
    
    def compute_homophily_loss(self, modified_features, adj_matrix):
        """计算同质性损失
        确保修改后节点特征与邻居特征分布相似
        L_homophily = sum_v ||f_v - mean(f_neighbors)||^2 for v in trigger_nodes
        """
        pass
    
    def compute_topology_stealth_loss(self, original_adj, modified_adj):
        """计算拓扑隐蔽性损失
        确保注入边的权重分布与原有边一致
        """
        pass
```

**关键实现要点**：
1. 特征触发器使用可学习的扰动向量，通过同质性约束调整方向
2. 拓扑触发器只在trigger_nodes和target_nodes之间添加边，权重采样自原有边权重分布
3. 使用mask确保只有trigger_nodes被修改

**输出物**：完整的`spatial_trigger.py`

### Task 3.3: 单元测试STM模块
**指令**：创建`tests/test_spatial_trigger.py`，验证STM模块功能：
1. 测试节点选择算法的输出
2. 测试特征触发器只修改指定节点
3. 测试拓扑触发器添加的边是否合理
4. 测试同质性损失的计算
5. 可视化修改前后的图结构差异（保存为图片）

**输出物**：`tests/test_spatial_trigger.py`，测试通过日志

---

## Phase 4: 协同优化器与注入引擎

### Task 4.1: 实现协同优化器
**指令**：在`cstba/modules/synergy_optimizer.py`中实现协同优化逻辑：

```python
class SynergyOptimizer:
    """协同优化器
    
    实现双层优化框架，联合优化TTM和STM参数
    """
    
    def __init__(self, ttm_module, stm_module, surrogate_model,
                 lambda_attack=1.0, lambda_stealth=0.5, lambda_synergy=0.3):
        """
        Args:
            ttm_module: 时序触发器模块实例
            stm_module: 空域触发器模块实例
            surrogate_model: 代理ST-GNN模型
            lambda_*: 各损失项权重
        """
        pass
    
    def compute_attack_loss(self, predictions, target_labels):
        """计算攻击损失
        对于回归任务：L_attack = -|prediction - target|
        对于分类任务：L_attack = CrossEntropy(prediction, target)
        """
        pass
    
    def compute_stealth_loss(self, ttm_module, stm_module, 
                              original_data, poisoned_data, adj_matrix):
        """计算综合隐蔽性损失
        L_stealth = L_smooth + L_freq + L_homophily + L_topology_stealth
        """
        pass
    
    def compute_synergy_loss(self, temporal_trigger, spatial_trigger, predictions):
        """计算协同损失
        使用MINE估计器计算条件互信息：-I(δ_t; δ_s | Y_target)
        互信息最大化促使两个触发器捕获互补信息
        """
        pass
    
    def inner_loop_update(self, poisoned_data, target_labels):
        """下层优化：更新代理模型参数
        θ* = argmin_θ L_train(f_θ(X_poison), Y_target)
        """
        pass
    
    def outer_loop_update(self, original_data, adj_matrix, target_labels):
        """上层优化：更新触发器生成器参数
        min_{ψ,φ} L_attack + λ₁·L_stealth + λ₂·L_synergy
        """
        pass
    
    def bilevel_optimize(self, train_loader, adj_matrix, target_labels,
                         num_iterations=100):
        """执行双层优化
        交替进行inner_loop和outer_loop更新
        """
        pass
```

**关键实现要点**：
1. 使用MINE (Mutual Information Neural Estimation) 估计互信息
2. inner_loop使用较大学习率快速收敛，outer_loop使用较小学习率精细调整
3. 记录优化过程中的各项损失变化用于分析

**输出物**：完整的`synergy_optimizer.py`

### Task 4.2: 实现触发注入引擎
**指令**：在`cstba/modules/trigger_injection.py`中实现数据投毒逻辑：

```python
class TriggerInjectionEngine:
    """触发注入引擎
    
    负责将生成的触发器植入训练/测试数据
    """
    
    def __init__(self, ttm_module, stm_module, poison_rate=0.1,
                 trigger_mode='joint'):
        """
        Args:
            ttm_module: 训练好的时序触发器模块
            stm_module: 训练好的空域触发器模块
            poison_rate: 投毒比例
            trigger_mode: 'temporal_only' / 'spatial_only' / 'joint'
        """
        pass
    
    def select_poison_samples(self, dataset_size, seed=None):
        """随机选择要投毒的样本索引"""
        pass
    
    def inject_temporal_trigger(self, x):
        """注入时序触发器
        Args:
            x: 原始数据 [B, T, V, C]
        Returns:
            x_poisoned: 注入触发器后的数据
        """
        pass
    
    def inject_spatial_trigger(self, x, adj_matrix):
        """注入空域触发器
        同时返回修改后的特征和邻接矩阵（如果是topology模式）
        """
        pass
    
    def inject_joint_trigger(self, x, adj_matrix):
        """联合注入时序和空域触发器"""
        pass
    
    def create_poisoned_dataset(self, clean_dataset, adj_matrix, 
                                 target_labels, save_path=None):
        """创建投毒数据集
        Args:
            clean_dataset: 原始干净数据集
            adj_matrix: 图邻接矩阵
            target_labels: 目标标签（用于标签翻转）
            save_path: 保存路径（可选）
        Returns:
            poisoned_dataset: 投毒后的数据集
            poison_indices: 被投毒的样本索引
        """
        pass
    
    def create_triggered_test_samples(self, test_data, adj_matrix):
        """创建测试时的触发样本（不修改标签）
        用于评估ASR
        """
        pass
```

**关键实现要点**：
1. 确保投毒样本的选择是可复现的（使用seed）
2. 对于回归任务，target_labels应该是一个极端值（如将犯罪率预测为很高）
3. 保存投毒信息以便后续分析

**输出物**：完整的`trigger_injection.py`

### Task 4.3: 实现主攻击类
**指令**：在`cstba/attacks/cstba_attack.py`中实现统一的攻击接口：

```python
class CSTBAAttack:
    """CSTBA攻击主类
    
    提供完整的攻击流程：触发器训练 -> 数据投毒 -> 后门模型训练 -> 评估
    """
    
    def __init__(self, config_path='cstba/config/cstba_config.yaml'):
        """加载配置并初始化各模块"""
        pass
    
    def analyze_graph(self, adj_matrix):
        """分析图结构，选择触发节点"""
        pass
    
    def train_trigger_generators(self, clean_train_loader, adj_matrix,
                                  target_nodes, target_labels):
        """训练触发器生成器（双层优化）"""
        pass
    
    def create_poisoned_data(self, clean_dataset, adj_matrix, target_labels):
        """创建投毒数据集"""
        pass
    
    def train_backdoor_model(self, model_class, poisoned_train_loader, 
                              clean_val_loader, adj_matrix, epochs, **kwargs):
        """训练后门模型"""
        pass
    
    def evaluate(self, backdoor_model, clean_test_loader, adj_matrix):
        """评估攻击效果
        返回：ASR (joint), ASR (temporal), ASR (spatial), BA Drop, Stealthiness
        """
        pass
    
    def run_full_attack(self, clean_dataset, adj_matrix, model_class,
                        target_nodes, target_labels, **kwargs):
        """执行完整攻击流程"""
        pass
    
    def save_attack_artifacts(self, save_dir):
        """保存攻击产物（触发器、投毒数据、后门模型）"""
        pass
```

**输出物**：完整的`cstba_attack.py`

---

## Phase 5: 训练与评估脚本

### Task 5.1: 实现后门模型训练脚本
**指令**：创建`cstba/scripts/train_backdoor.py`：

```python
"""
后门模型训练脚本

使用方式：
python -m cstba.scripts.train_backdoor \
    --config cstba/config/cstba_config.yaml \
    --model gwn \
    --dataset chicago_crime \
    --target_regions 10,15,23 \
    --output_dir experiments/cstba_gwn_chicago
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='CSTBA Backdoor Training')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['stgcn', 'dcrnn', 'gwn', 'astgcn'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--target_regions', type=str, help='逗号分隔的目标区域节点ID')
    parser.add_argument('--output_dir', type=str, default='experiments/')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载配置和数据
    # 2. 初始化CSTBAAttack
    # 3. 分析图结构，选择触发节点
    # 4. 训练触发器生成器
    # 5. 创建投毒数据集
    # 6. 训练后门模型
    # 7. 保存模型和攻击产物
    
    pass

if __name__ == '__main__':
    main()
```

脚本应该输出详细的训练日志，包括每个epoch的损失变化。

**输出物**：完整的`train_backdoor.py`

### Task 5.2: 实现攻击评估脚本
**指令**：创建`cstba/scripts/evaluate_attack.py`：

```python
"""
攻击效果评估脚本

使用方式：
python -m cstba.scripts.evaluate_attack \
    --model_path experiments/cstba_gwn_chicago/backdoor_model.pth \
    --trigger_path experiments/cstba_gwn_chicago/triggers/ \
    --dataset chicago_crime \
    --output_dir experiments/cstba_gwn_chicago/evaluation
"""

def main():
    # 1. 加载后门模型和触发器
    # 2. 加载测试数据
    # 3. 评估三种触发模式的ASR
    # 4. 评估BA Drop
    # 5. 评估隐蔽性指标
    # 6. 生成评估报告
    # 7. 生成可视化图表
    
    pass
```

评估报告应包含：
- ASR对比表（Temporal-Only vs Spatial-Only vs Joint）
- BA Drop随投毒率变化曲线
- 触发器可视化（时序波形、空间节点分布）
- 预测偏移热力图（显示哪些区域被成功攻击）

**输出物**：完整的`evaluate_attack.py`

### Task 5.3: 创建实验运行脚本
**指令**：创建`run_experiments.sh`批量运行实验：

```bash
#!/bin/bash
# CSTBA完整实验脚本

# 实验1: 不同模型对比
for model in stgcn dcrnn gwn astgcn; do
    python -m cstba.scripts.train_backdoor \
        --config cstba/config/cstba_config.yaml \
        --model $model \
        --dataset chicago_crime \
        --target_regions 10,15,23 \
        --output_dir experiments/model_comparison/${model}
done

# 实验2: 不同投毒率对比
for rate in 0.05 0.08 0.10 0.12 0.15; do
    # 修改配置中的poison_rate并运行
    ...
done

# 实验3: 消融实验（Temporal-Only, Spatial-Only, Joint）
for mode in temporal_only spatial_only joint; do
    ...
done

# 评估所有实验
python -m cstba.scripts.evaluate_attack --batch experiments/
```

**输出物**：`run_experiments.sh`

---

## Phase 6: 集成测试与文档

### Task 6.1: 端到端集成测试
**指令**：创建`tests/test_integration.py`，验证完整攻击流程：

1. 使用小规模合成数据测试完整流程
2. 验证以下关键指标：
   - 触发器生成成功
   - 投毒数据集创建成功
   - 后门模型训练收敛
   - ASR (Joint) > ASR (Temporal) 且 ASR (Joint) > ASR (Spatial)
   - BA Drop < 1%

**输出物**：`tests/test_integration.py`，集成测试通过日志

### Task 6.2: 编写使用文档
**指令**：创建`cstba/README.md`，包含：

1. 算法原理简介（引用设计文档）
2. 安装依赖说明
3. 快速开始示例
4. 配置参数详解
5. API文档（主要类和函数说明）
6. 实验复现指南
7. 常见问题解答

**输出物**：`cstba/README.md`

### Task 6.3: 代码质量检查
**指令**：执行代码质量检查和优化：

1. 运行pylint或flake8检查代码规范
2. 添加类型注解（type hints）
3. 确保所有函数有docstring
4. 优化明显的性能瓶颈
5. 确保GPU加速正确使用（如果可用）

**输出物**：代码质量报告，优化后的代码

---

## 执行顺序与依赖关系

```
Phase 0 (环境准备)
    │
    ▼
Phase 1 (数据结构)
    │
    ├──────────────┬──────────────┐
    ▼              ▼              │
Phase 2 (TTM)   Phase 3 (STM)    │
    │              │              │
    └──────┬───────┘              │
           ▼                      │
    Phase 4 (协同优化)  ◄─────────┘
           │
           ▼
    Phase 5 (训练评估)
           │
           ▼
    Phase 6 (集成测试)
```

---

## 关键检查点

**Checkpoint 1** (Phase 1完成后)：验证项目结构正确，配置可加载，指标函数可运行

**Checkpoint 2** (Phase 2完成后)：TTM模块单独可用，能生成合理的时序触发器

**Checkpoint 3** (Phase 3完成后)：STM模块单独可用，节点选择合理，特征触发器有效

**Checkpoint 4** (Phase 4完成后)：协同优化器能收敛，联合触发效果优于单独触发

**Checkpoint 5** (Phase 5完成后)：完整攻击流程可运行，输出符合预期

**Checkpoint 6** (Phase 6完成后)：所有测试通过，文档完善，代码质量达标

---

## 注意事项

1. **代码复用**：尽量复用现有代码库中的数据加载、模型定义和训练循环逻辑，只在必要时修改
2. **设备兼容**：确保代码同时支持CPU和GPU运行
3. **随机种子**：所有随机操作都应支持seed设置以确保可复现
4. **内存管理**：处理大规模图数据时注意内存使用，必要时使用mini-batch
5. **日志记录**：使用logging模块记录关键信息，便于调试
6. **版本控制**：每完成一个Task后commit一次，commit message应清晰描述变更内容

---

## 预期产出物清单

| Phase | 关键产出文件 |
|-------|------------|
| 0 | `docs/project_structure_analysis.md`, `docs/dataset_format.md` |
| 1 | `cstba/` 目录结构, `cstba_config.yaml`, `metrics.py` |
| 2 | `temporal_trigger.py`, `frequency_utils.py`, `test_temporal_trigger.py` |
| 3 | `node_selection.py`, `spatial_trigger.py`, `test_spatial_trigger.py` |
| 4 | `synergy_optimizer.py`, `trigger_injection.py`, `cstba_attack.py` |
| 5 | `train_backdoor.py`, `evaluate_attack.py`, `run_experiments.sh` |
| 6 | `test_integration.py`, `cstba/README.md`, 代码质量报告 |
