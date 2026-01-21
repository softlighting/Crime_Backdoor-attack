import React, { useState } from 'react';

const CSTBAArchitecture = () => {
  const [activeTab, setActiveTab] = useState('architecture');
  const [highlightModule, setHighlightModule] = useState(null);

  const ModuleBox = ({ title, subtitle, color, children, id, className = "" }) => (
    <div 
      className={`rounded-lg border-2 p-4 transition-all duration-300 ${className}`}
      style={{ 
        borderColor: color,
        backgroundColor: highlightModule === id ? `${color}20` : 'white',
        transform: highlightModule === id ? 'scale(1.02)' : 'scale(1)'
      }}
      onMouseEnter={() => setHighlightModule(id)}
      onMouseLeave={() => setHighlightModule(null)}
    >
      <div className="font-bold text-sm" style={{ color }}>{title}</div>
      {subtitle && <div className="text-xs text-gray-500 mb-2">{subtitle}</div>}
      {children}
    </div>
  );

  const Arrow = ({ direction = "down", className = "" }) => (
    <div className={`flex justify-center ${className}`}>
      {direction === "down" && <div className="text-gray-400 text-2xl">↓</div>}
      {direction === "right" && <div className="text-gray-400 text-2xl">→</div>}
      {direction === "both" && <div className="text-gray-400 text-xl">⟷</div>}
    </div>
  );

  const ArchitectureView = () => (
    <div className="space-y-4">
      {/* Input Layer */}
      <div className="grid grid-cols-2 gap-4">
        <ModuleBox title="原始时空数据" subtitle="X ∈ ℝ^(T×V×C)" color="#3B82F6" id="input-x">
          <div className="text-xs text-gray-600 mt-1">
            时间步长 T、节点数 V、特征维度 C
          </div>
        </ModuleBox>
        <ModuleBox title="原始图结构" subtitle="A ∈ ℝ^(V×V)" color="#3B82F6" id="input-a">
          <div className="text-xs text-gray-600 mt-1">
            邻接矩阵、边权重
          </div>
        </ModuleBox>
      </div>

      <Arrow direction="down" />

      {/* Trigger Modules */}
      <div className="grid grid-cols-2 gap-4">
        <ModuleBox title="时序触发器模块 (TTM)" subtitle="Temporal Trigger Module" color="#10B981" id="ttm">
          <div className="space-y-2 mt-2">
            <div className="bg-green-50 rounded p-2 text-xs">
              <div className="font-semibold text-green-700">GAT生成器 Gᵩ</div>
              <div className="text-gray-600">多变量耦合建模</div>
            </div>
            <div className="bg-green-50 rounded p-2 text-xs">
              <div className="font-semibold text-green-700">形状感知归一化</div>
              <div className="text-gray-600">频域隐形约束</div>
            </div>
            <div className="bg-green-50 rounded p-2 text-xs">
              <div className="font-semibold text-green-700">时间平滑损失</div>
              <div className="text-gray-600">连续性保证</div>
            </div>
          </div>
        </ModuleBox>
        <ModuleBox title="空域触发器模块 (STM)" subtitle="Spatial Trigger Module" color="#8B5CF6" id="stm">
          <div className="space-y-2 mt-2">
            <div className="bg-purple-50 rounded p-2 text-xs">
              <div className="font-semibold text-purple-700">关键节点识别</div>
              <div className="text-gray-600">中心性融合得分</div>
            </div>
            <div className="bg-purple-50 rounded p-2 text-xs">
              <div className="font-semibold text-purple-700">特征触发器 Gφ</div>
              <div className="text-gray-600">同质性伪装</div>
            </div>
            <div className="bg-purple-50 rounded p-2 text-xs">
              <div className="font-semibold text-purple-700">拓扑触发器</div>
              <div className="text-gray-600">虚拟传播路径</div>
            </div>
          </div>
        </ModuleBox>
      </div>

      {/* Synergy Optimizer */}
      <div className="flex items-center gap-2">
        <div className="flex-1 h-px bg-gray-300"></div>
        <ModuleBox title="协同优化器" subtitle="Synergy Optimizer" color="#F59E0B" id="synergy" className="flex-shrink-0">
          <div className="text-xs text-gray-600 mt-1 space-y-1">
            <div>ℒ_attack + λ₁·ℒ_stealth + λ₂·ℒ_synergy</div>
            <div className="text-amber-600 font-semibold">互信息最大化</div>
          </div>
        </ModuleBox>
        <div className="flex-1 h-px bg-gray-300"></div>
      </div>

      <Arrow direction="down" />

      {/* Injection Engine */}
      <ModuleBox title="触发注入引擎" subtitle="Trigger Injection Engine" color="#EF4444" id="injection">
        <div className="grid grid-cols-3 gap-2 mt-2">
          <div className="bg-red-50 rounded p-2 text-xs text-center">
            <div className="font-semibold text-red-700">训练投毒</div>
            <div className="text-gray-600">5%-15%样本</div>
          </div>
          <div className="bg-red-50 rounded p-2 text-xs text-center">
            <div className="font-semibold text-red-700">标签翻转</div>
            <div className="text-gray-600">Y → Y_target</div>
          </div>
          <div className="bg-red-50 rounded p-2 text-xs text-center">
            <div className="font-semibold text-red-700">稀疏注入</div>
            <div className="text-gray-600">关键节点+时间窗</div>
          </div>
        </div>
      </ModuleBox>

      <Arrow direction="down" />

      {/* Output */}
      <ModuleBox title="后门ST-GNN模型" subtitle="Backdoored Model fθ" color="#1F2937" id="output">
        <div className="text-xs text-gray-600 mt-1">
          ST-GCN / DCRNN / Graph WaveNet / ASTGCN
        </div>
      </ModuleBox>
    </div>
  );

  const ActivationView = () => (
    <div className="space-y-6">
      <div className="text-center text-sm font-semibold text-gray-700 mb-4">
        推理阶段触发激活模式对比
      </div>
      
      {/* Mode A: Temporal Only */}
      <div className="flex items-center gap-4 p-4 bg-green-50 rounded-lg">
        <div className="w-24 text-center">
          <div className="text-xs font-bold text-green-700">模式 A</div>
          <div className="text-xs text-gray-600">仅时序触发</div>
        </div>
        <div className="flex-1 flex items-center gap-2">
          <div className="bg-white rounded px-3 py-2 text-xs border">测试样本</div>
          <span className="text-green-500">→</span>
          <div className="bg-green-100 rounded px-3 py-2 text-xs border border-green-300">
            TTM激活<br/>周期模式
          </div>
          <span className="text-green-500">→</span>
          <div className="bg-yellow-100 rounded px-3 py-2 text-xs border border-yellow-300">
            部分偏移<br/>ASR ~65%
          </div>
        </div>
      </div>

      {/* Mode B: Spatial Only */}
      <div className="flex items-center gap-4 p-4 bg-purple-50 rounded-lg">
        <div className="w-24 text-center">
          <div className="text-xs font-bold text-purple-700">模式 B</div>
          <div className="text-xs text-gray-600">仅空域触发</div>
        </div>
        <div className="flex-1 flex items-center gap-2">
          <div className="bg-white rounded px-3 py-2 text-xs border">测试样本</div>
          <span className="text-purple-500">→</span>
          <div className="bg-purple-100 rounded px-3 py-2 text-xs border border-purple-300">
            STM激活<br/>节点特征
          </div>
          <span className="text-purple-500">→</span>
          <div className="bg-yellow-100 rounded px-3 py-2 text-xs border border-yellow-300">
            部分偏移<br/>ASR ~75%
          </div>
        </div>
      </div>

      {/* Mode C: Joint */}
      <div className="flex items-center gap-4 p-4 bg-gradient-to-r from-green-50 to-purple-50 rounded-lg border-2 border-amber-300">
        <div className="w-24 text-center">
          <div className="text-xs font-bold text-amber-700">模式 C</div>
          <div className="text-xs text-gray-600">联合触发</div>
        </div>
        <div className="flex-1 flex items-center gap-2">
          <div className="bg-white rounded px-3 py-2 text-xs border">测试样本</div>
          <span className="text-gray-500">→</span>
          <div className="flex flex-col gap-1">
            <div className="bg-green-100 rounded px-2 py-1 text-xs border border-green-300">TTM</div>
            <div className="bg-purple-100 rounded px-2 py-1 text-xs border border-purple-300">STM</div>
          </div>
          <span className="text-amber-500">→</span>
          <div className="bg-amber-100 rounded px-3 py-2 text-xs border border-amber-400 font-semibold">
            协同放大<br/>1+1 {'>'} 2
          </div>
          <span className="text-red-500">→</span>
          <div className="bg-red-100 rounded px-3 py-2 text-xs border border-red-400 font-bold text-red-700">
            恶意预测<br/>ASR ~95%
          </div>
        </div>
      </div>

      <div className="mt-4 p-3 bg-gray-50 rounded text-xs text-gray-600">
        <span className="font-semibold">协同优于单独的机制：</span>
        互信息正则化确保TTM和STM捕获互补信息，级联放大效应通过空域修改的消息传递路径增强时序信号传播。
      </div>
    </div>
  );

  const PropagationView = () => (
    <div className="space-y-4">
      <div className="text-center text-sm font-semibold text-gray-700 mb-4">
        时空协同传播机制
      </div>
      
      <div className="relative bg-gray-50 rounded-lg p-6">
        {/* Time axis */}
        <div className="flex justify-between items-center mb-2">
          <div className="text-xs text-gray-500">t-T+1</div>
          <div className="text-xs text-gray-500">t-T/2</div>
          <div className="text-xs text-gray-500">t-T/3</div>
          <div className="text-xs font-bold text-green-600">TTM注入开始</div>
          <div className="text-xs text-gray-500">t</div>
        </div>
        <div className="h-1 bg-gradient-to-r from-gray-300 via-green-300 to-green-500 rounded mb-4"></div>

        {/* Spatial grid */}
        <div className="grid grid-cols-5 gap-2">
          {/* Row 1: Trigger Node v1 */}
          <div className="text-right text-xs text-gray-500 flex items-center justify-end">触发节点 v1</div>
          {[0,1,2,3,4].map(i => (
            <div key={`v1-${i}`} className={`h-10 rounded flex items-center justify-center text-xs font-bold ${i >= 2 ? 'bg-green-400 text-white' : 'bg-gray-200 text-gray-600'}`}>
              {i >= 2 ? '●' : '○'}
            </div>
          ))}
          
          {/* Row 2: Relay Node v2 */}
          <div className="text-right text-xs text-gray-500 flex items-center justify-end">中继节点 v2</div>
          {[0,1,2,3,4].map(i => (
            <div key={`v2-${i}`} className={`h-10 rounded flex items-center justify-center text-xs font-bold ${i >= 3 ? 'bg-purple-400 text-white' : 'bg-gray-200 text-gray-600'}`}>
              {i >= 3 ? '●' : '○'}
            </div>
          ))}
          
          {/* Row 3: Target Region */}
          <div className="text-right text-xs text-gray-500 flex items-center justify-end">目标区域</div>
          {[0,1,2,3,4].map(i => (
            <div key={`target-${i}`} className={`h-10 rounded flex items-center justify-center text-xs font-bold ${i === 4 ? 'bg-red-500 text-white animate-pulse' : 'bg-gray-200 text-gray-600'}`}>
              {i === 4 ? '★' : '○'}
            </div>
          ))}
        </div>

        {/* Arrows showing propagation */}
        <svg className="absolute top-0 left-0 w-full h-full pointer-events-none" style={{zIndex: 10}}>
          {/* Temporal propagation arrow */}
          <defs>
            <marker id="arrowhead-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#10B981" />
            </marker>
            <marker id="arrowhead-purple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#8B5CF6" />
            </marker>
          </defs>
        </svg>

        {/* Legend */}
        <div className="flex gap-4 justify-center mt-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 bg-gray-200 rounded"></div>
            <span>正常状态</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 bg-green-400 rounded"></div>
            <span>时序触发信号</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 bg-purple-400 rounded"></div>
            <span>空域传播信号</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span>后门激活</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="p-3 bg-green-50 rounded text-xs">
          <div className="font-bold text-green-700 mb-1">时序传播 (TTM)</div>
          <div className="text-gray-600">
            在时间窗口后1/3注入周期微振荡模式，沿时间轴向前传播，被TCN/GRU捕获并记忆
          </div>
        </div>
        <div className="p-3 bg-purple-50 rounded text-xs">
          <div className="font-bold text-purple-700 mb-1">空域传播 (STM)</div>
          <div className="text-gray-600">
            通过关键节点的特征修改，利用GNN消息传递机制向目标区域扩散触发信号
          </div>
        </div>
      </div>
    </div>
  );

  const ModelAdaptView = () => (
    <div className="space-y-4">
      <div className="text-center text-sm font-semibold text-gray-700 mb-4">
        模型适配策略
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 rounded-lg border-2 border-blue-300 bg-blue-50">
          <div className="font-bold text-blue-700 text-sm">ST-GCN</div>
          <div className="text-xs text-gray-600 mt-2 space-y-1">
            <div><span className="font-semibold">架构：</span>图卷积 + 时间卷积解耦</div>
            <div><span className="font-semibold">TTM策略：</span>周期模式适配卷积核</div>
            <div><span className="font-semibold">STM策略：</span>特征注入利用平滑传播</div>
            <div className="text-blue-600 font-semibold">推荐投毒率：8%-12%</div>
          </div>
        </div>

        <div className="p-4 rounded-lg border-2 border-teal-300 bg-teal-50">
          <div className="font-bold text-teal-700 text-sm">DCRNN</div>
          <div className="text-xs text-gray-600 mt-2 space-y-1">
            <div><span className="font-semibold">架构：</span>扩散卷积 + GRU门控</div>
            <div><span className="font-semibold">TTM策略：</span>突变-恢复激活更新门</div>
            <div><span className="font-semibold">STM策略：</span>双向可达性高的节点</div>
            <div className="text-teal-600 font-semibold">推荐投毒率：6%-10%</div>
          </div>
        </div>

        <div className="p-4 rounded-lg border-2 border-amber-400 bg-amber-50">
          <div className="font-bold text-amber-700 text-sm">Graph WaveNet ⭐</div>
          <div className="text-xs text-gray-600 mt-2 space-y-1">
            <div><span className="font-semibold">架构：</span>自适应邻接矩阵 + 扩散 + TCN</div>
            <div><span className="font-semibold">TTM策略：</span>触发节点相似模式诱导聚合</div>
            <div><span className="font-semibold">STM策略：</span>利用自适应矩阵放大效应</div>
            <div className="text-amber-600 font-semibold">推荐投毒率：5%-8% (最佳目标)</div>
          </div>
        </div>

        <div className="p-4 rounded-lg border-2 border-pink-300 bg-pink-50">
          <div className="font-bold text-pink-700 text-sm">ASTGCN</div>
          <div className="text-xs text-gray-600 mt-2 space-y-1">
            <div><span className="font-semibold">架构：</span>空间注意力 + 时间注意力</div>
            <div><span className="font-semibold">TTM策略：</span>分布式片段适应稀疏选择</div>
            <div><span className="font-semibold">STM策略：</span>高注意力节点特征注入</div>
            <div className="text-pink-600 font-semibold">推荐投毒率：7%-11%</div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white min-h-screen">
      <div className="text-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">协同式时空后门触发器注入算法</h1>
        <h2 className="text-lg text-gray-500">Collaborative Spatio-Temporal Backdoor Trigger Algorithm (CSTBA)</h2>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b mb-6">
        {[
          { id: 'architecture', label: '系统架构' },
          { id: 'activation', label: '激活机制' },
          { id: 'propagation', label: '时空传播' },
          { id: 'models', label: '模型适配' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab.id 
                ? 'border-b-2 border-blue-500 text-blue-600' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="bg-white rounded-lg">
        {activeTab === 'architecture' && <ArchitectureView />}
        {activeTab === 'activation' && <ActivationView />}
        {activeTab === 'propagation' && <PropagationView />}
        {activeTab === 'models' && <ModelAdaptView />}
      </div>

      {/* Performance Table */}
      <div className="mt-8 p-4 bg-gray-50 rounded-lg">
        <div className="text-sm font-semibold text-gray-700 mb-3">预期性能指标</div>
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-gray-200">
              <th className="p-2 text-left">触发模式</th>
              <th className="p-2 text-center">ST-GCN</th>
              <th className="p-2 text-center">DCRNN</th>
              <th className="p-2 text-center">GWN</th>
              <th className="p-2 text-center">ASTGCN</th>
              <th className="p-2 text-center">BA Drop</th>
            </tr>
          </thead>
          <tbody>
            <tr className="bg-green-50">
              <td className="p-2">Temporal-Only</td>
              <td className="p-2 text-center">62%</td>
              <td className="p-2 text-center">68%</td>
              <td className="p-2 text-center">71%</td>
              <td className="p-2 text-center">65%</td>
              <td className="p-2 text-center text-green-600">&lt;0.5%</td>
            </tr>
            <tr className="bg-purple-50">
              <td className="p-2">Spatial-Only</td>
              <td className="p-2 text-center">73%</td>
              <td className="p-2 text-center">76%</td>
              <td className="p-2 text-center">82%</td>
              <td className="p-2 text-center">75%</td>
              <td className="p-2 text-center text-green-600">&lt;0.7%</td>
            </tr>
            <tr className="bg-amber-100 font-bold">
              <td className="p-2">Joint (CSTBA)</td>
              <td className="p-2 text-center text-red-600">91%</td>
              <td className="p-2 text-center text-red-600">93%</td>
              <td className="p-2 text-center text-red-600">97%</td>
              <td className="p-2 text-center text-red-600">92%</td>
              <td className="p-2 text-center text-green-600">&lt;1.0%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default CSTBAArchitecture;
