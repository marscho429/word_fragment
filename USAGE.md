# 蛋白词-分子片段规则优化系统使用说明

## 系统概述

本系统实现了蛋白词-分子片段的泛化规则生成，能够为每个分子片段对应的多个蛋白词提取高质量的泛化规则。

## 模块结构

### 1. core.py - 核心数据结构
- 定义了氨基酸性质分类（TIERED_SEMANTIC_GROUPS）
- 实现了规则元素类型（RuleElement）和泛化规则（GeneralizationRule）
- 提供了辅助函数（如性质类查找、元素创建等）

### 2. preprocess.py - 预处理模块
- 处理蛋白词中的下划线
- 计算氨基酸序列重叠度
- 按长度分组、聚类等

### 3. generalize.py - 泛化规则生成
- 多序列比对算法
- 规则生成策略（分层挖掘）
- 覆盖率优化

### 4. validate.py - 规则验证
- 规则匹配验证
- FPR计算
- 质量评估

### 5. pipeline.py - 完整处理流程
- 数据加载
- 单fragment处理
- 结果输出

## 快速开始

### 1. 环境准备
```bash
cd /DATA/DATANAS1/marszhao-srt/rules
# 使用rules_env环境
/DATA/DATANAS1/marszhao-srt/rules_env/bin/python3.10
```

### 2. 基本功能测试
```bash
python test_basic.py
```

### 3. 小规模数据测试
```bash
python test_pipeline_small.py
```

### 4. 运行完整pipeline
```bash
cd /DATA/DATANAS1/marszhao-srt/rules/src
python pipeline.py --max_fragments 10 --output_dir results
```

## 主要参数

- `--data_dir`: 数据目录（默认：dataset）
- `--output_dir`: 输出目录（默认：results）
- `--max_time`: 单个fragment最大处理时间（默认：60秒）
- `--target_coverage`: 目标覆盖率（默认：0.85）
- `--max_fpr`: 最大假阳性率（默认：0.1）
- `--sample_size`: 负样本池大小（默认：10000）
- `--max_fragments`: 最多处理的fragment数量（用于测试）

## 输出结果

### pipeline_results.json
包含整体统计信息：
- 总fragments数量
- 成功fragments数量
- 平均覆盖率
- 平均FPR
- 总规则数
- 处理时间

### detailed_rules.pkl
包含每个fragment的详细规则：
- Pattern: 规则表达式
- Covered_Words: 覆盖的蛋白词列表
- Covered_Count: 覆盖数量
- Average_Score: 平均score
- False_Positive_Rate: FPR值

## 规则语法

系统支持以下规则元素：

1. **确切氨基酸**: `A` - 单个氨基酸
2. **性质类**: `[acidic]` - 氨基酸性质分类
   - Tier 1: acidic, basic, sulfur, amide, hydroxyl, tiny, aromatic
   - Tier 2: aliphatic, cyclic, uncharged_polar
   - Tier 3: small, nonpolar, polar, bulky
3. **可选类**: `(A)` - 可选氨基酸
4. **任意氨基酸**: 
   - `x(3)` - 固定数量
   - `x(1,3)` - 范围数量
   - `_` - 一个或多个

## 当前实现状态

### 已完成
✅ 核心数据结构定义
✅ 氨基酸性质分类
✅ 蛋白词预处理（包括下划线处理）
✅ 重叠度计算
✅ 多序列比对（基础版本）
✅ 规则生成策略（分层挖掘）
✅ 规则验证和FPR计算
✅ 完整的pipeline流程

### 需要优化
⚠️ 比对算法：当前使用简单的滑动窗口方法，覆盖率较低
⚠️ 规则生成策略：需要更智能的聚类和合并策略
⚠️ 性能优化：对于大规模数据（3746个fragments），需要进一步优化

### 测试结果
- 基本功能测试：全部通过 ✅
- 小规模数据测试：可运行，但覆盖率较低（约30%）⚠️

## 使用示例

### 示例1：创建简单规则
```python
from core import create_exact_element, create_property_element, GeneralizationRule

# 创建元素
elem1 = create_exact_element('K')  # 确切氨基酸
elem2 = create_property_element('basic')  # 性质类

# 创建规则
rule = GeneralizationRule(
    elements=[elem1, elem2],
    covered_words={'KYRG', 'KR'}
)
```

### 示例2：处理蛋白词
```python
from preprocess import preprocess_word

# 处理带下划线的蛋白词
word = preprocess_word('L_VIS_DR')
print(word.amino_sequence)  # ['L', 'V', 'I', 'S', 'D', 'R']
print(word.has_underscore)  # True
```

### 示例3：生成规则
```python
from generalize import generate_rules_for_fragment

# 为fragment生成规则
rules = generate_rules_for_fragment(
    words=['RYRGSDC', 'KYRGGSDC'],
    word_scores={'RYRGSDC': 0.5, 'KYRGGSDC': 0.6},
    max_time=10.0
)
```

## 下一步工作

1. **优化比对算法**：
   - 使用更高级的多序列比对方法（如Needleman-Wunsch或Smith-Waterman）
   - 改进gap处理策略

2. **改进规则生成策略**：
   - 使用更智能的聚类算法（如层次聚类）
   - 增加规则合并和优化步骤

3. **性能优化**：
   - 使用并行处理
   - 优化内存使用
   - 减少不必要的计算

4. **扩展功能**：
   - 支持更多规则类型
   - 添加可视化功能
   - 提供更多统计指标

## 注意事项

- 当前实现的覆盖率较低，不适合直接用于生产环境
- 建议先用小规模数据测试（max_fragments参数）
- 输出的规则需要进一步验证和质量评估