# 蛋白词-分子片段规则优化方案

环境：micromamba activate rules_env即可激活
# 1.数据处理
数据集在/DATA/DATANAS1/marszhao-srt/rules/dataset下面（写代码的时候使用相对路径），底下包含3个pickle文件，下面我先说明一下三个pickle文件的内容：
### 1.1:每个分子片段的信息
frag_library_v1.pkl
这也是dict，记录了片段id对应的具体结构。key是片段的ID，value中包含了片段的smarts表达式，以及用rdikt得到的pattern对象。
前2个键值对如下：
  键 : frag_1  ==>  值: {'Smarts': '[NH1R0](-[*])-[*]', 'Pattern': <rdkit.Chem.rdchem.Mol object at 0x7f0f6d32fa60>, 'ID': 'frag_1', 'Frequency': {'total_affinity_smiles': 954981}, 'Frag from': ['total_affinity_smiles_macfrag_with_broken']}
  键 : frag_2  ==>  值: {'Smarts': '[OH0R0]=[CH0R0](-[*])-[*]', 'Pattern': <rdkit.Chem.rdchem.Mol object at 0x7f0f6d32fb00>, 'ID': 'frag_2', 'Frequency': {'total_affinity_smiles': 1071343}, 'Frag from': ['total_affinity_smiles_macfrag_with_broken']}

### 1.2:每个蛋白词与哪些分子片段是配对的（没啥用？）
pwfm_hard_1024_1024_processed_ww2_word2frags.pkl
这是一个dict，key是蛋白词-分子片段对，value是这个对的score。score越高表示模型的置信度越高
前2个键值对如下：
  键 : IATLTG frag_3217  ==>  值: 0.472716413302993
  键 : IATLTG frag_827  ==>  值: 0.41926203502905834

### 1.3:蛋白词-分子片段对,score分数越高模型的置信度越高
pwfm_hard_1024_1024_processed_ww2_wordfrag2score.pkl
这也是一个dict，key是蛋白词，value是分子片段ID的集合，表示该蛋白词与集合中的每一个片段都存在配对规则。内容其实和上一个pkl是一样的，只是这个文件没记录score
前2个键值对如下：
  键 : IATLTG  ==>  值: {'frag_2565', 'frag_143', 'frag_34', 'frag_263', 'frag_259', 'frag_2042', 'frag_260', 'frag_747', 'frag_851', 'frag_827', 'frag_1370', 'frag_1161', 'frag_4715', 'frag_3419', 'frag_203', 'frag_4654', 'frag_1158', 'frag_3659', 'frag_4990', 'frag_1915', 'frag_3520', 'frag_1698', 'frag_3217', 'frag_5148', 'frag_2313', 'frag_4939', 'frag_551', 'frag_2502', 'frag_5058', 'frag_820', 'frag_2403'}
  键 : L_LVGKGITFDSGGYNLKA  ==>  值: {'frag_3217', 'frag_3219', 'frag_2210', 'frag_746', 'frag_747', 'frag_4129', 'frag_151', 'frag_1738', 'frag_2660'}

*特别注意，这里的下划线可以代表一个或多个氨基酸残基，你把它看作一种特殊的片段即可*
*然后我注意到大部分蛋白词都是有下划线的，一般一个蛋白词会有0～2个下划线*
*输入的蛋白词只包含具体的氨基酸和下划线，而输出的泛化规则包含的内容会多一些，下面会具体定义

* 第一步是把蛋白词-与之配对的分子片段的词典转化成分子片段-与之配对的蛋白词的词典，存入一个origin.pkl中（已经做好了，直接调用就可以）

## 2.具体聚类规则的实现
* 对于每一个分子片段，把与之配对的蛋白词“合并”，以下是具体规则
## 2.1定义总结出来规则的语法
A          固定氨基酸
[ABC]      三选一
x          任意氨基酸
x(n)       n个任意氨基酸
x(m,n)     m~n个任意氨基酸
_          下划线表示一个或多个任意氨基酸
(A)        可选A
[Aromatic] 性质类
例如：[VTA]?YRGSDC 或者[KR]-x(1,3)-GD

关于性质类我认为最关键的是这几类
```python
TIERED_SEMANTIC_GROUPS = [
    # ---------------------------------------------------------
    # Tier 1: 绝对高优组 (极度特异，2~3个元素)
    # 一旦命中，必须无条件优先提取！
    # ---------------------------------------------------------
    ('acidic', {'D', 'E'}, 1),                 # 酸性/带负电
    ('basic', {'K', 'R', 'H'}, 1),             # 碱性/带正电
    ('sulfur', {'C', 'M'}, 1),                 # 含硫基团
    ('amide', {'N', 'Q'}, 1),                  # 酰胺基
    ('hydroxyl', {'S', 'T', 'Y'}, 1),          # 羟基 (常见磷酸化位点)
    ('tiny', {'G', 'A', 'S'}, 1),              # 极小体积 (转角核心)
    ('aromatic', {'F', 'Y', 'W'}, 2),          # 芳香族 (含苯环)
    # ---------------------------------------------------------
    # Tier 2: 中优先级 (特定骨架特征，4~6个元素)
    # Tier 1 不匹配时，退而求其次
    # ---------------------------------------------------------
    ('aliphatic', {'A', 'V', 'L', 'I'}, 2),    # 脂肪族链
    ('cyclic', {'P', 'F', 'Y', 'W', 'H'}, 2),  # 环状结构
    ('uncharged_polar', {'S', 'T', 'N', 'Q', 'Y', 'C'}, 2), # 不带电极性

    # ---------------------------------------------------------
    # Tier 3: 兜底低优组 (宽泛分类，8~11个元素)
    # 只有在万不得已，又要为了避免变成 x 时才使用
    # ---------------------------------------------------------
    ('small', {'G', 'A', 'S', 'C', 'D', 'P', 'N', 'T'}, 3),          # 小体积
    ('nonpolar', {'A', 'V', 'L', 'I', 'P', 'F', 'M', 'W', 'G'}, 3),  # 疏水大类
    ('polar', {'S', 'T', 'N', 'Q', 'Y', 'C', 'D', 'E', 'K', 'R', 'H'}, 3), # 极性大类
    ('bulky', {'F', 'Y', 'W', 'R', 'K', 'E', 'Q', 'M', 'L', 'I'}, 3) # 大体积
]
```
## 3.任务要求
你需要为每个分子片段（Fragment）对应的多个蛋白词（Words）提取高质量的泛化规则，
比如该分子片段对应的蛋白词有RYRGSDC和KYRGGSDC,那泛化规则可以是[basic]YRGSDC
又比如YRGSDC和DRWYRGSDCRTI,那泛化规则可以是x(0,3)-YRGSDC-x(0,3)
以上两个例子是我认为比较典型也是我最希望看到的
下面详细讲述规则：
### 3.1 什么样的规则是好规则
* 必须满足的硬性标准：
  - 1.泛化后的规则必须能匹配原先的对应的蛋白词，反面例子比如x(0,2)-IVII-x(0,3)，而原始的词是G_I_VII，因为_代表了至少一个氨基酸，所以泛化后的规则无法匹配原始的氨基酸
  - 2.总共有3746个`fragments`，每个`fragment`对应多个蛋白词，一个蛋白词也对应多个fragments，比如说当前`fragment`总结出了100条泛化规则，那我用没有在当前fragment里面出现的蛋白词（比如我采样10000个，当然如果时间消耗过长的话可以少采样一点），我希望它们都不符合这100条泛化规则，也就是这10000个蛋白词都是负样本，而它的假阳性率FPR必须维持一个很低的水平，这里假设是0.1（注意，你对于每条泛化规则检测FPR然后累加是没有用的，因为可能有重复的导致FPR虚高，你只能针对这100条规则的总体来检验）
  - 比如当前fragment中有1000个蛋白词，其中900个蛋白词都被我用于总结泛化后的规则了（每条泛化后的规则至少能匹配两个蛋白词），那么泛化覆盖率就是900/1000=0.9，说明我总结出的泛化规则能匹配90%的蛋白词，我希望泛化覆盖率维持在较高的水平，至少不要低于85%
* 一些软性标准
  - 泛化规则中，只能出现四种内容，确切的氨基酸、性质类（如[acidic]）、可选类（如(A)）和任意氨基酸（如x(0,3),x(n)）
重要性排序：确切的氨基酸>>性质类>>可选类(比如ABCD和ABD可以总结为AB(C)D)>任意氨基酸
  - 我们希望每条规则中任意氨基酸的比例不超过50%，否则规则过于宽泛，FPR很难维持较低水平
  - 性质类有三类优先级，请积极查表（非常重要！！），多去提取性质类而不是任意氨基酸
  - 我们希望每条泛化后的规则能覆盖2～5个蛋白词，否则规则会过于宽泛，FPR很难维持较低水平
  - 我希望的一种模式是，你可以先拿算法去筛选一下那些有很高氨基酸重叠度的规则，比如[acidic]YRGSDC和KYRGSDC这样的，这种是最理想的规则肯定要尽量保留；确定找不到这种最理想的规则后，把它们排除掉，然后再去找次理想的规则，也就是更关注每个氨基酸后面所属的大类，然后把性质类也加入到规则中（当然，如果一个位置几个规则都是相同氨基酸的话优先选这个氨基酸），另外如果有些位置中间可能就隔了一两个氨基酸比较碍事的话可以用(A)这样的符号来规避一下。我想的就是按照规则的好坏这么一层层的筛，筛到你看这个泛化覆盖率已经超过85%或者90%了你再停止，后面的规则才保留原始规则



## 3.3输出与性能规范
**JSON 输出 Schema**
最终生成的 JSON 结果中，每条规则对象必须记录以下完整信息：
* `Pattern`: 具体的规则表述（如 `[basic]-x(1,3)-G-D`）。
* `Covered_Words`: 该规则实际覆盖的原始蛋白词列表。
* `Covered_Count`: 覆盖的蛋白词数量。
* `Average_Score`: 该规则覆盖的这几个“蛋白词-分子片段”组合，在原始数据中的`score`的平均值。
每个fragment必须记录的信息：
* `False_Positive_Rate`: 使用上述目标2中定义的“负样本池”实际打靶计算出的单条规则 FPR 值。
总体

**7. 性能约束 (Time Complexity)**
* 避免 $O(N^2)$ 的深层嵌套死循环和无意义的全库正则扫街。
* 目前有3746个fragments，每个fragment对应几百至几千个蛋白词，请合理使用算法，使得时间满足要求
* 要求单个分子片段的挖掘时间控制在 **1 分钟以内**，理想状态为 **30 秒以内**完成。


### 4.算法设计

我想的是采用**基于相似度的成对聚类（有点像自底而上的凝聚式聚类，但是细节上不太一样）**策略，为每个分子片段生成高质量的泛化规则。核心思想是：对蛋白词进行两两比对，根据相似度构建稀疏图，然后贪心地合并高相似度的词对，生成覆盖多个词的共识规则。
因为python写的话实在太慢了，所以用c++实现了

#### 4.1 核心数据结构

**Token**：规则中的基本单元，分为以下类型：
- 字面量（Literal）：如 `A`、`G`
- 性质类（Semantic Class）：如 `[acidic]`、`[aliphatic]`，带有优先级（Tier 1-3）
- 可选（Optional）：如 `(A)`，表示该氨基酸可省略
- 通配符（Wildcard）：如 `x(1,3)`、`_`（在这里代表1-4个任意氨基酸）

**PairEdge（词对边）**：存储两个蛋白词的比对结果
```cpp
struct PairEdge {
    int i, j;                    // 词对索引
    double score;                // 比对得分
    std::string rule;            // 生成的规则字符串
    std::vector<Token> tokens;   // 规则令牌列表
    double complexity;           // 规则复杂度
};
```

**Cluster（簇）**：一组蛋白词及其共识规则
```cpp
struct Cluster {
    int id;
    std::vector<int> word_indices;  // 包含的词索引
    std::string rule;                // 共识规则
    std::vector<Token> tokens;       // 规则令牌
    double complexity;              // 复杂度
    int center_word_idx;             // 中心词索引
    bool active;                     // 是否活跃
};
```

#### 4.2 动态规划比对（半全局比对）

对于两个蛋白词 `w1` 和 `w2`，使用动态规划计算最优比对：

**状态定义**：`dp[i][j]` 表示 `w1[0..i-1]` 与 `w2[0..j-1]` 的最优比对得分

**转移操作**：
1. **匹配**：相同字符 +1.0，氨基酸性质类匹配按层级加分（Tier1=0.9, Tier2=0.7, Tier3=0.5），不同字符 +0.3
2. **删除（Optional）**：跳过 `w1` 中的字符，罚分 -0.5
3. **插入（侧翼）**：在 `w1` 后添加侧翼通配符 `x(0,N)`，加分 `-0.1*N`
4. **下划线匹配**：`_` 可匹配 1-4 个字符，加分 `0.5*k`

**终止条件**：采用半全局比对，允许 `w1` 末尾自由延长侧翼

#### 4.3 氨基酸性质类分层

按优先级分为三层，优先使用高特异性的性质类：

| Tier | 性质类 | 氨基酸集合 | 权重 |
|------|--------|-----------|------|
| 1 | acidic | DE | 0.9 |
| 1 | basic | KRH | 0.9 |
| 1 | sulfur | CM | 0.9 |
| 1 | amide | NQ | 0.9 |
| 1 | hydroxyl | STY | 0.9 |
| 1 | tiny | GAS | 0.9 |
| 2 | aromatic | FYW | 0.7 |
| 2 | aliphatic | AVLI | 0.7 |
| 2 | cyclic | PFYWH | 0.7 |
| 3 | small | GASCDPNT | 0.5 |
| 3 | nonpolar | AVLIPFMWG | 0.5 |
| 3 | polar | STNQYCDERKHR | 0.5 |
| 3 | bulky | FYW RKEQMLI | 0.5 |

#### 4.4 算法流程

**Phase 1：计算所有词对比对得分**
```
for each pair (i, j) where i < j:
    align = DP_align(words[i], words[j])
    if align.score >= PAIR_THRESHOLD (2.0):
        edge = PairEdge(i, j, align)
        adj[i].push_back(edge)
        adj[j].push_back(edge)
```
时间复杂度：$O(N^2 \cdot L^2)$，其中 $N$ 为词数，$L$ 为平均词长

**Phase 2：TopK 剪枝**
```
for each word i:
    keep top TOPK_PER_WORD (1000000) edges by score
```
取消 TopK 限制（设为极大值），保留所有超过阈值的边

**Phase 3：初始化簇**
```
for each word i:
    clusters[i] = Cluster(i, {i}, words[i])
```

**Phase 4：基于词对的聚类**
```
sorted_edges = sort all edges by score descending
for each edge (i, j) in sorted_edges:
    if clustered[i] or clustered[j]: continue
    if edge.score < PAIR_THRESHOLD: break
    
    // 快速路径：使用缓存的比对结果
    tokens = edge.tokens
    if rule_matches_word(tokens, words[i]) and rule_matches_word(tokens, words[j]):
        if validate_constraints(tokens):
            merge(i, j)
            continue
    
    // 尝试泛化
    tokens = generalize(tokens, uncovered_word)
    if validate_constraints(tokens):
        merge(i, j)
        continue
    
    // 回退：重新计算比对
    tokens = build_consensus_rule({i, j})
    if success: merge(i, j)
```

**Phase 5-6：规则扫描与 FPR 计算**
```
for each cluster:
    compute pos_bits: which words this rule covers
    
for each cluster and each neg sample:
    compute neg_bits: which negatives this rule covers
    
overall_fpr = |union of neg_bits| / |negative_pool|
```

**Phase 7：贪心规则选择**
```
candidates = sort clusters by (coverage desc, complexity asc)

for each candidate in candidates:
    if new_coverage == 0: continue
    if new_fpr > max_fpr: continue  // 跳过会导致FPR超限的规则
    
    select this rule
    covered |= pos_bits
```

**Phase 8：添加精确匹配规则**
```
for each uncovered word:
    add exact-match rule (word itself)
```

#### 4.5 规则约束

生成的规则必须满足以下约束：
- **核心位置数** ≥ 3（`MIN_CORE_POSITIONS = 3`）
- **最大单通配符长度** ≤ 4（`MAX_SINGLE_WILDCARD = 4`）
- **通配符总长度** ≤ 10（`MAX_WILDCARD_TOTAL = 10`）
- **Tier3 性质类数量** ≤ 2（`MAX_TIER3_COUNT = 2`）
- **复杂度** ≤ 15（`MAX_COMPLEXITY = 15`）

复杂度计算：
```cpp
double compute_rule_complexity(const std::vector<Token>& tokens) {
    double cx = 0;
    for (auto& t : tokens) {
        if (t.type == LITERAL) cx += 1.0;
        else if (t.type == SEMANTIC_CLASS) cx += (t.tier == 1 ? 0.9 : t.tier == 2 ? 0.7 : 0.5);
        else if (t.type == OPTIONAL) cx += 0.3;
        else if (t.type == WILDCARD) cx += t.max_len * 0.1;
    }
    return cx;
}
```

#### 4.6 规则泛化

当初始比对规则不能覆盖某个词时，尝试泛化令牌：
```
token -> [semantic_class] -> x(min_len, max_len)
```
泛化策略：
1. 字面量优先尝试转换为性质类（按 Tier 顺序）
2. 性质类尝试扩展为通配符
3. 记录泛化历史，避免重复泛化

#### 4.7 性能优化

1. **缓存对齐结果**：Phase 1 计算的对齐结果缓存在 `PairEdge` 中，Phase 4 直接复用
2. **Float 替代 Double**：DP 矩阵使用 `float`，减少内存占用和计算时间
3. **稀疏图剪枝**：只保留得分 ≥ 2.0 的边
4. **Bitset 优化**：规则覆盖使用 `std::bitset<MAX_WORDS>`，FPR 计算使用位运算

#### 4.8 覆盖率定义

- **总体覆盖率**：所有规则覆盖的词数 / 总词数（= 100%，因为每个词至少有一条规则）
- **泛化覆盖率**：多词规则（覆盖 ≥ 2 个词）覆盖的词数 / 总词数

根据实验观测，泛化覆盖率受限于数据本身的结构：**约 30-35% 的词可以形成可泛化的聚类**，剩余 65-70% 的词两两之间相似度不足，难以生成有效的泛化规则。这是数据特性决定的，而非算法问题。


### 5.输出结果
Total Time: 29672.47s (7.92s per fragment)
Total Words: 2523287
Covered Words: 2523287
Overall Coverage: 100.00%
Gen-Covered Words: 845998
Overall Gen-Coverage: 33.53%
Total Rules: 2095274
* 我预想的gen-coverage能达到90%以上，但是实际只有33.53%，我调整了很久，但是好像用不同算法算出来的如果泛化规则的覆盖率高了，那么FPR就会很高，这个规则质量就会很低，可能这说明确实存在很大一部分的规则是没办法被泛化的

### 6.运行方式

#### 6.1 环境配置
```bash
# 激活 conda 环境（路径需根据实际安装位置调整）
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate rules
```

#### 6.2 编译 C++ 模块（首次运行或代码修改后）
```bash
# 进入项目根目录
cd rules

# 编译 C++ 模块
c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) $(python3-config --includes --ldflags) src/rule_generator.cpp -o src/rule_generator_cpp.cpython-311-x86_64-linux-gnu.so
```

#### 6.3 运行方式

**测试运行（前3个fragment）：**
```bash
cd rules
python3 run_rules.py --num-fragments 3 --output outputs/rules_test.json
```

**全量运行（所有3746个fragment）：**
```bash
cd rules
python3 run_rules.py --output outputs/rules_ahc_full.json
```

**自定义参数：**
```bash
cd rules

# 指定目标覆盖率和最大FPR
python3 run_rules.py --target-coverage 0.85 --max-fpr 0.1 --output outputs/rules_custom.json

# 指定负样本池大小
python3 run_rules.py --negative-sample-size 1000 --output outputs/rules_custom.json
```

#### 6.4 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | str | `dataset/origin.pkl` | 输入数据文件路径 |
| `--output` | str | `outputs/rules_ahc.json` | 输出结果文件路径 |
| `--num-fragments` | int | None | 处理的fragment数量（None=全部） |
| `--target-coverage` | float | 0.85 | 目标覆盖率 |
| `--max-fpr` | float | 0.1 | 最大假阳性率 |
| `--negative-sample-size` | int | 1000 | 负样本池大小 |

#### 6.5 输出结果

运行结束后，输出文件为 JSON 格式，包含每个 fragment 的规则列表和统计信息：

```json
{
  "frag_150": {
    "Fragment_ID": "frag_150",
    "Total_Words": 869,
    "Rules": [
      {
        "Pattern": "[basic]-Y-R-G-S-D-C",
        "Covered_Words": ["KYRGSDC", "RYRGSDC"],
        "Covered_Count": 2,
        "Average_Score": 0.45
      }
    ],
    "Coverage": 1.0,
    "Gen_Coverage": 0.3579,
    "Overall_FPR": 0.098
  }
}
```

#### 6.6 处理大文件（Git 提交）

生成的规则文件可能非常大（全量约 GB 级别），无法直接 push 到 Git 仓库。建议：

1. **忽略 outputs 目录**（已配置在 `.gitignore`）：
   ```bash
   # .gitignore 内容
   outputs/
   __pycache__/
   *.pyc
   *.so
   ```

2. **仅提交源代码**：
   ```bash
   git add .gitignore README.md run_rules.py requirements.txt src/rule_generator.cpp
   git commit -m "Add rule generation algorithm"
   git push
   ```

3. **规则文件存储建议**：
   - 将生成的规则文件存储在服务器本地或云存储（如 NAS、S3）
   - 不建议将大输出文件纳入版本控制

4. **运行时间预估**：
   - 每个 fragment 约 8-12 秒
   - 全量 3746 个 fragment 约需 8-12 小时
