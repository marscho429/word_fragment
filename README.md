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

* 第一步是把蛋白词-与之配对的分子片段的词典转化成分子片段-与之配对的蛋白词的词典，存入一个origin.pkl中

## 2.具体聚类规则的实现
* 对于每一个分子片段，把与之配对的蛋白词“合并”，以下是具体规则
## 2.1定义总结出来规则的语法
A          固定氨基酸
[ABC]      三选一
x          任意氨基酸
x(n)       n个任意氨基酸
x(m,n)     m~n个任意氨基酸
(A)        可选A
[Aromatic] 性质类
例如：[VTA]?YRGSDC 或者[KR]-x(1,3)-GD

## 2.2先把所有蛋白词转成序列
比如A_VRGSDC变成["A","_","V","R","G","S","D","C"]
这里"_"是特殊token，后面会变成x(1,n)

# 🧬 蛋白词泛化规则挖掘算法：基于信息熵与 Set Cover 的全局寻优

## 1. 核心理论重构与预处理

### 1.1 真正的负样本定义 (True Negatives)
放弃完全随机打乱的乱码（可作为极小比例的辅助），改用具有真实物理意义的背景分布。
* **全库蛋白词 (Global_Words)**: 整个数据集中出现过的所有去重蛋白词集合。
* **正样本 ($Pos\_Pool_{frag}$)**: 当前分子片段（如 `frag_1`）已知匹配的蛋白词。
* **负样本 ($Neg\_Pool_{frag}$)**: $Global\_Words - Pos\_Pool_{frag}$。
*(逻辑：在自然界中存在，但明确没有出现在当前 fragment 结合列表中的词，才是最高质量的“硬负样本 / Hard Negatives”)*。

### 1.2 信息熵与列确凿度 (Shannon Entropy & IC Score)
抛弃拍脑袋的打分。每一列的保守程度用标准的香农熵计算。
已知自然界有 20 种标准氨基酸，完美保守的列信息量为 $\log_2(20) \approx 4.32$ bits。

* **列熵计算公式**: 
  $$Entropy = -\sum_{i=1}^{n} p_i \log_2(p_i)$$
  *(其中 $p_i$ 是该列某氨基酸出现的频率)*
* **确凿信息度 (Information Content, IC)**: 
  $$IC = 4.32 - Entropy$$
  *(例如：全是 `Y`，Entropy=0，IC=4.32；一半 `D` 一半 `E`，Entropy=1，IC=3.32)*。

### 1.3 下划线 `_` 的柔性保留策略
不再武断地将 `_` 转换为 `x(1,3)`。在对齐和聚类阶段，将 `_` 统一映射为特殊的占位符 `<GAP>`。最终输出规则时，保留为类似 `[DE]-<GAP>-Y-R-G` 的格式，待后续引入真实三维结构数据时，再反算其物理距离区间。

---

## 2. 核心数学定义与数据结构

### 2.1 样本空间定义 (Sample Spaces)
为避免多重药理学（交叉结合）带来的假阴性惩罚，严格区分正样本与背景分布：
* **全局词库 ($U$)**: 数据集中所有去重后的蛋白词序列集合。
* **正样本集合 ($P$)**: 当前特定分子片段（Fragment）已知结合的去重蛋白词集合。
* **背景分布池 ($B$)**: $B = U \setminus P$。
> **注**：$B$ 代表 Unlabeled Data，用于估计规则的全局泛化能力与假阳性率（FPR），防止规则过度泛化。

### 2.2 确凿信息度 (Information Content, IC)
用于衡量提取出的“实心块”的保守程度，规避毫无规律的随机突变。
* **香农熵公式**: 
  $$Entropy = -\sum_{i=1}^{20} p_i \log_2(p_i)$$
  *(算法实现约束：需过滤占位符 `_`，并对 $p_i = 0$ 的项进行条件拦截，防止除零错误)*
* **IC 得分**: 
  $$IC\_Score = 4.32 - Entropy$$
  *(设定硬性底线 `MIN_IC_SCORE = 3.0`，低于此值的块将被抛弃)*

### 2.3 先验理化字典 (Prior Knowledge Base)
在计算出高质量的局部块后，若存在同族变异，使用此字典进行状态简并：
```python
AMINO_ACID_GROUPS = {
    '[DE]': {'D', 'E'}, '[KRH]': {'K', 'R', 'H'}, '[FYW]': {'F', 'Y', 'W'}, 
    '[LIVM]': {'L', 'I', 'V', 'M'}, '[ST]': {'S', 'T'}, '[NQ]': {'N', 'Q'}, 
    '[GAS]': {'G', 'A', 'S'}
}
```
---
## 3. 算法主流程 (The Algorithm Pipeline)

### Phase 1: 高判别力锚点挖掘 (Discriminative Anchor Extraction)
**目标**：寻找在正样本中高度富集，而在全局背景中极其罕见的核心保守区（Seed）。

1. **K-mers 提取**：扫描正样本 $P$ 中所有序列，提取长度 $\ge 3$ 且不含 `_` 的实心子串集合 $K$。
2. **频率惩罚打分 (TF-IDF 思路)**：对每个 $k \in K$ 计算锚点得分，压制高频无意义片段（如 `AAA`）：
   $$AnchorScore = \log_2(Support_P + 1) \times (-\log_2(Frequency_U))$$
   *(其中 $Support_P$ 为 $k$ 在正样本中的出现次数，$Frequency_U$ 为其在全局库中的频率)*
3. **选种**：按 $AnchorScore$ 降序排列，取 Top-N（如前 20 个）作为**候选种子池 (Seed_Pool)**。

### Phase 2: 基于滑动窗口的局部块延展 (Local-Block Expansion)
**目标**：抛弃死板的按列扫描，采用“滑动窗口找下一个高优支点”的策略解决 `_` 导致的错位问题。

针对每一个 Seed，执行以下探索：
1. **起点锚定**：提取 $P$ 中包含该 Seed 的序列集合，以 Seed 为 `Current_Anchor` 对齐。
2. **向右滑动搜索 (Rightward Search)**：
   * 设定物理滑动窗口 `W`（如向右 1~4 个氨基酸跨度）。
   * 在窗口内穷举寻找下一个使 $IC\_Score$ 最大化的单列字母或同族简并块 (`Next_Block`)。
   * **熔断**：若窗口内找不到任何 $IC\_Score \ge 3.0$ 的块，说明进入极度柔性区，向右延展终止。
3. **动态 Gap 结算**：
   * 统计各序列中 `Current_Anchor` 与 `Next_Block` 之间的实际氨基酸物理距离，求得该区间的 `min_d` 和 `max_d`。
   * 拼接出带弹性区间的临时规则：`Temp_Rule = Current_Anchor + <GAP(min_d, max_d)> + Next_Block`。
4. **正则编译与打靶校验 (Regex Evaluation)**：
   * 将 `Temp_Rule` 编译为正则表达式（如 `<GAP(1,3)>` 映射为 `.{1,3}`）。
   * 扫描背景池 $B$ 计算 $FPR$。若 $FPR > Max\_FPR$（如 1%），则撤销本次延展，终止向右探索；若合格，则将 `Next_Block` 更新为新的 `Current_Anchor`，继续向右搜索。
5. **向左滑动搜索 (Leftward Search)**：逻辑同上，方向相反。
6. **保存结果**：双向延展停止后，将该 Seed 对应的局部最优规则加入**候选规则集 ($C_{rules}$)**。

### Phase 3: 集合覆盖全局寻优 (Set Cover Approximation)
**目标**：解决贪婪偏见 (Greedy Bias)，从 $C_{rules}$ 中挑选最少且最优的规则组合，最大化覆盖正样本。

1. **初始化**：未覆盖集合 $P_{uncovered} = P$，最终规则集 $R_{final} = \emptyset$。
2. **动态评估与挑选**：
   **WHILE** $P_{uncovered}$ 非空：
   * 对 $C_{rules}$ 中每条规则，重新计算在 $P_{uncovered}$ 中的**有效新增覆盖数** ($\Delta Coverage$)。若全员 $\Delta Coverage = 0$ 则跳出循环。
   * 将指标进行 Min-Max 归一化至 $[0, 1]$，计算综合效用：
     $$Utility = w_1 \cdot \Delta Coverage_{norm} + w_2 \cdot IC_{norm} - w_3 \cdot FPR_{norm}$$
   * 选取 $Utility$ 最高的规则移入 $R_{final}$，并从 $P_{uncovered}$ 中永久剔除其覆盖的序列。

### Phase 4: 置信度聚合与指标输出
对于 $R_{final}$ 中的每一条泛化规则：
1. **聚合置信度打分**：获取该规则成功覆盖的所有原始正样本的匹配 Score，采用 **10% 截尾平均数 (Trimmed Mean)** 或中位数计算该规则的最终 Score，消除极值干扰。
2. **压缩率计算**：$Compression\_Ratio = |P| / |R_{final}|$。

## 4. 正则表达式编译引擎规范 (Evaluation Engine)
在评估规则覆盖率和 FPR 时，必须将提取出的 PROSITE 语法转化为计算机可识别的正则表达式 (Regex)：
* 固定字母：`A` $\rightarrow$ `A`
* 简并家族：`[DE]` $\rightarrow$ `[DE]`
* 通配符：`x` $\rightarrow$ `.`
* 弹性区间：`<GAP(m, n)>` $\rightarrow$ `.{m,n}` (注：Python 实现中应按需使用非贪婪模式 `.{m,n}?`)

---

## 5. 输出数据结构规范 (Output Schema)
最终以 JSON 或 Dict 形式输出每个 Fragment 的挖掘结果：
```json
{
  "Fragment_ID": "frag_3217",
  "Total_Original_Words": 150,
  "Compression_Ratio": 30.0, 
  "Rules": [
    {
      "Pattern": "[FYW]-R-G-<GAP(0,2)>-D-C",
      "Average_Score": 0.88,
      "Covered_Count": 85,
      "Rule_IC_Score": 23.4,
      "False_Positive_Rate": 0.002
    },
    {
      "Pattern": "[LIVM]-<GAP(1,3)>-T-L-A",
      "Average_Score": 0.72,
      "Covered_Count": 65,
      "Rule_IC_Score": 18.1,
      "False_Positive_Rate": 0.005
    }
  ]
}
