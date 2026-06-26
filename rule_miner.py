#!/usr/bin/env python3
"""
蛋白词-分子片段规则挖掘算法 (v13.0)
严格遵守钢铁纪律：
1. 动态局部简并（Dynamic Brackets）取代死板降级
2. 严禁内部连续通配符（Anti-String-of-Pearls）
3. 最低有效位占比（Minimum Informative Positions >= 70%）

用法: python rule_miner.py [--first N]
"""

import pickle
import re
import math
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import random

import numpy as np

# ============================================================
# 常量定义
# ============================================================

AA_LETTERS = set('ACDEFGHIKLMNPQRSTVWY')
# 铁律1：下划线是合法字符，代表柔性loop
AA_LETTERS_WITH_UNDERSCORE = set('ACDEFGHIKLMNPQRSTVWY_')

AMINO_ACID_GROUPS = {
    '[DE]': {'D', 'E'},
    '[KRH]': {'K', 'R', 'H'},
    '[FYW]': {'F', 'Y', 'W'},
    '[LIVM]': {'L', 'I', 'V', 'M'},
    '[ST]': {'S', 'T'},
    '[NQ]': {'N', 'Q'},
    '[GAS]': {'G', 'A', 'S'},
}

SEMANTIC_GROUPS = [
    ('acidic',   {'D', 'E'}),
    ('basic',    {'K', 'R', 'H'}),
    ('aromatic', {'F', 'Y', 'W'}),
    ('aliphatic',{'L', 'I', 'V', 'M'}),
    ('hydroxyl', {'S', 'T'}),
    ('amide',    {'N', 'Q'}),
    ('small',    {'G', 'A', 'S'}),
]

SEMANTIC_TO_REGEX = {name: '[' + ''.join(sorted(aas)) + ']' for name, aas in SEMANTIC_GROUPS}

MAX_FPR = 0.005       # 最大假阳性率
MAX_GAP_SPAN = 3      # Gap跨度限制
MAX_GAP_UPPER = 3     # 单个Gap最大上限
MAX_WILDCARD_RATIO = 0.5  # 最大通配比例
FPR_SAMPLE_SIZE = 5000   # 采样大小

# 纪律1：动态局部简并阈值
MAX_BRACKET_SIZE = 3      # 方括号内最多3种氨基酸，超过则降级为x
MIN_INFORMATIVE_RATIO = 0.70  # 纪律3：最低有效位占比70%

# 纪律2：严禁内部连续通配符
MAX_CONSECUTIVE_X = 1     # 核心区内部最多连续1个x

# 修复2：长度悬殊限制
MAX_LENGTH_DIFF = 5       # 合并序列长度差异超过5则拒绝

# ============================================================
# 铁律1：数据加载与下划线保留
# ============================================================

def load_data(dataset_dir='dataset'):
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / 'origin.pkl', 'rb') as f:
        frag2words = pickle.load(f)
    with open(dataset_dir / 'pwfm_hard_1024_1024_processed_ww2_word2frags.pkl', 'rb') as f:
        word2frags = pickle.load(f)
    with open(dataset_dir / 'pwfm_hard_1024_1024_processed_ww2_wordfrag2score.pkl', 'rb') as f:
        wordfrag2score = pickle.load(f)
    global_words = set(word2frags.keys())
    return frag2words, word2frags, wordfrag2score, global_words


# ============================================================
# 铁律1：保留下划线的序列处理
# ============================================================

# 全局缓存
global_aa_seqs = {}


def set_global_aa_seqs(word_set):
    """
    设置全局氨基酸序列缓存
    铁律1：必须保留下划线 `_`
    """
    global global_aa_seqs
    # 保留下划线！下划线代表柔性loop
    global_aa_seqs = {w: ''.join(c for c in w if c in AA_LETTERS_WITH_UNDERSCORE) for w in word_set}


def get_full_sequence(word):
    """
    从缓存获取完整序列（保留下划线）
    铁律1：绝不能擦除下划线
    """
    return global_aa_seqs.get(word, ''.join(c for c in word if c in AA_LETTERS_WITH_UNDERSCORE))


def get_solid_sequence(word):
    """
    获取实心氨基酸序列（擦除下划线，用于某些特殊计算）
    注意：只在必要时使用，大部分情况应该用 get_full_sequence
    """
    return ''.join(c for c in word if c in AA_LETTERS)


def get_aa_group(aa):
    """获取氨基酸所属的家族集合"""
    for group_set in AMINO_ACID_GROUPS.values():
        if aa in group_set:
            return group_set
    return set()


def _aa_set_to_block(aa_set):
    """将氨基酸集合转换为语义化表示"""
    if len(aa_set) == 1:
        return list(aa_set)[0]

    for name, group_set in SEMANTIC_GROUPS:
        if aa_set.issubset(group_set):
            return f'[{name}]'

    sorted_aas = ''.join(sorted(aa_set))
    return f'[{sorted_aas}]'


# ============================================================
# 铁律4：严格的规则转换（PROSITE语法）
# ============================================================

def rule_to_regex(rule_pattern):
    """
    将规则模式转换为正则表达式
    铁律4：严格处理下划线，下划线映射为任意氨基酸
    """
    parts = rule_pattern.split('-')
    regex_parts = []
    for part in parts:
        if part.startswith('x(') and part.endswith(')'):
            inner = part[2:-1]
            if ',' in inner:
                m, n = inner.split(',')
                m, n = int(m), int(n)
            else:
                m = n = int(inner)
            # 铁律4：gap区间必须匹配下划线（下划线代表柔性loop）
            regex_parts.append(f'[A-Z_]{{{m},{n}}}')
        elif part.startswith('<GAP(') and part.endswith(')>'):
            inner = part[5:-2]
            m, n = inner.split(',')
            m, n = int(m), int(n)
            regex_parts.append(f'[A-Z_]{{{m},{n}}}')
        elif part == 'x':
            regex_parts.append('[A-Z_]')
        elif part == '_':
            # 铁律1：下划线代表柔性loop，匹配任意氨基酸
            regex_parts.append('[A-Z_]')
        else:
            regex_parts.append(block_to_regex(part))
    return ''.join(regex_parts)


def block_to_regex(block):
    """将一个block转换为正则表达式片段"""
    if block.startswith('[') and block.endswith(']'):
        inner = block[1:-1]
        if inner in SEMANTIC_TO_REGEX:
            return SEMANTIC_TO_REGEX[inner]
        return block
    if len(block) == 1 and block in AA_LETTERS:
        return block
    # 铁律1：如果block包含下划线，作为通配符处理
    if block == '_':
        return '[A-Z_]'
    return block


# ============================================================
# 模块1：种子优先级排序（保留下划线）
# ============================================================

def sort_sequences_by_centrality(seq_list):
    """
    按中心度得分降序排列序列
    铁律1：计算3-mer时使用完整序列（保留下划线）
    """
    # 铁律1：使用完整序列，保留下划线
    full_seqs = [get_full_sequence(seq) for seq in seq_list]

    # 统计所有3-mer频次（保留下划线）
    kmer_counts = defaultdict(int)
    for full_seq in full_seqs:
        # 提取实心3-mer（忽略下划线，但计算位置时保留下划线）
        solid = get_solid_sequence(full_seq)
        for i in range(len(solid) - 2):
            kmer = solid[i:i+3]
            kmer_counts[kmer] += 1

    # 计算每条序列的中心度得分
    seq_scores = []
    for seq, full_seq in zip(seq_list, full_seqs):
        score = 0
        solid = get_solid_sequence(full_seq)
        for i in range(len(solid) - 2):
            kmer = solid[i:i+3]
            score += kmer_counts[kmer]
        seq_scores.append((seq, score))

    # 降序排列
    seq_scores.sort(key=lambda x: x[1], reverse=True)
    return [seq for seq, _ in seq_scores]


# ============================================================
# 模块2：严苛的规则校验器（铁律2 + 铁律3）
# ============================================================

def is_rule_valid(rule_pattern, background_pool):
    """
    严苛的规则校验（纪律2 + 纪律3 + FPR）

    纪律2：严禁内部连续通配符
    纪律3：最低有效位占比>=70%
    """
    parts = rule_pattern.split('-')
    
    # 识别核心区（剥离头尾GAP）
    core_start = 0
    core_end = len(parts)
    
    # 找到第一个非GAP的位置
    for i, part in enumerate(parts):
        if not (part.startswith('x(') or part.startswith('<GAP(') or part == 'x' or part == '_'):
            core_start = i
            break
    
    # 找到最后一个非GAP的位置
    for i in range(len(parts) - 1, -1, -1):
        if not (parts[i].startswith('x(') or parts[i].startswith('<GAP(') or parts[i] == 'x' or parts[i] == '_'):
            core_end = i + 1
            break
    
    # 提取核心区
    core_parts = parts[core_start:core_end]
    
    if len(core_parts) == 0:
        return False
    
    # 纪律2：检查内部连续通配符
    consecutive_x = 0
    for part in core_parts:
        if part.startswith('x(') or part == 'x' or part == '_':
            consecutive_x += 1
            # 纪律2：核心区内部连续x超过1，熔断
            if consecutive_x > MAX_CONSECUTIVE_X:
                return False
        else:
            consecutive_x = 0
    
    # 纪律3：计算有效位占比
    informative_count = 0
    for part in core_parts:
        if part.startswith('x(') or part == 'x' or part == '_':
            continue
        # 方括号内氨基酸数量<=3也算有效位
        if part.startswith('[') and part.endswith(']'):
            # 提取方括号内的氨基酸
            inner = part[1:-1]
            if len(inner) <= MAX_BRACKET_SIZE:
                informative_count += 1
            else:
                # 方括号内超过3种氨基酸，不算有效位
                continue
        elif len(part) == 1 and part in AA_LETTERS:
            informative_count += 1
    
    total_core = len(core_parts)
    if total_core == 0:
        return False
    
    informative_ratio = informative_count / total_core
    # 纪律3：有效位占比必须>=70%
    if informative_ratio < MIN_INFORMATIVE_RATIO:
        return False
    
    # Gap跨度限制
    for part in parts:
        if part.startswith('x(') and part.endswith(')'):
            inner = part[2:-1]
            if ',' in inner:
                m, n = inner.split(',')
                m, n = int(m), int(n)
                if n > MAX_GAP_UPPER or (n - m) > MAX_GAP_SPAN:
                    return False
        elif part.startswith('<GAP(') and part.endswith(')>'):
            inner = part[5:-2]
            m, n = inner.split(',')
            m, n = int(m), int(n)
            if n > MAX_GAP_UPPER or (n - m) > MAX_GAP_SPAN:
                return False
    
    # FPR检查（使用fullmatch）
    regex_str = rule_to_regex(rule_pattern)
    compiled = re.compile(regex_str)
    
    bg_total = len(background_pool)
    if bg_total == 0:
        return False
    
    # 采样优化
    bg_list = list(background_pool)
    if len(bg_list) > FPR_SAMPLE_SIZE:
        bg_list = random.sample(bg_list, FPR_SAMPLE_SIZE)
    
    # 早停机制
    max_fp = int(MAX_FPR * len(bg_list))
    bg_count = 0
    
    for word in bg_list:
        full_seq = get_full_sequence(word)
        if compiled.fullmatch(full_seq):
            bg_count += 1
            if bg_count > max_fp:
                return False
    
    return True


# ============================================================
# 模块3：多序列动态融合（纪律1+2+3）
# ============================================================

def merge_rule_and_sequences(covered_seqs):
    """
    多序列全局比对融合（纪律1：动态局部简并）
    
    核心改进：
    - 对所有序列进行全局比对（不是找公共子串）
    - 对每一列统计实际氨基酸，生成[AV]等精准简并
    - 只有>3种氨基酸才拒绝融合
    """
    if len(covered_seqs) < 2:
        return None
    
    # 纪律1：获取完整序列
    full_seqs = [get_full_sequence(seq) for seq in covered_seqs]
    
    # 对所有序列进行全局比对
    aligned_positions = align_multiple_sequences(full_seqs)
    
    if aligned_positions is None:
        return None
    
    # 纪律1：构建动态简并核心区
    core_parts = build_dynamic_degenerate_core(full_seqs, aligned_positions)
    
    if core_parts is None:
        return None
    
    # 计算左右gap
    left_lengths = [pos[0] if isinstance(pos, tuple) else pos for pos in aligned_positions]
    left_min = min(left_lengths)
    left_max = max(left_lengths)
    
    right_lengths = []
    for seq, pos in zip(full_seqs, aligned_positions):
        if isinstance(pos, tuple):
            end_pos = pos[1]
        else:
            end_pos = pos + len(core_parts)
        right_lengths.append(len(seq) - end_pos)
    
    right_min = min(right_lengths)
    right_max = max(right_lengths)
    
    # 构建完整规则
    parts = []
    
    # 左侧gap
    if left_max > 0:
        left_max = min(left_max, MAX_GAP_UPPER)
        left_min = min(left_min, left_max)
        if left_min == left_max:
            parts.append(f'x({left_min})')
        else:
            parts.append(f'x({left_min},{left_max})')
    
    # 核心区
    parts.extend(core_parts)
    
    # 右侧gap
    if right_max > 0:
        right_max = min(right_max, MAX_GAP_UPPER)
        right_min = min(right_min, right_max)
        if right_min == right_max:
            parts.append(f'x({right_min})')
        else:
            parts.append(f'x({right_min},{right_max})')
    
    return '-'.join(parts)


def align_multiple_sequences_anchored(seqs):
    """
    多序列锚点比对（修复1：交集而非并集，同时保留动态方括号）
    
    核心思想：
    1. 找到所有序列共有的最长简并锚点（允许每列2-3种氨基酸）
    2. 锚点本身就是动态简并核心区
    3. 锚点两侧的区域用gap表示
    
    返回：(aligned_parts, left_min, left_max, right_min, right_max)
    """
    if len(seqs) < 2:
        return None
    
    # 找所有序列共有的最长简并锚点（直接返回核心区）
    result = find_best_anchor_for_multiple(seqs)
    
    if result is None:
        return None
    
    anchor_positions, core_parts = result
    
    if len(core_parts) < 2:
        return None
    
    # 纪律3：检查有效位占比
    informative_count = sum(1 for part in core_parts if '[' in part or (len(part) == 1 and part in AA_LETTERS))
    if informative_count < len(core_parts) * MIN_INFORMATIVE_RATIO:
        return None
    
    # 计算gap区间（修复1：交集模式）
    left_lengths = [pos for pos in anchor_positions]
    left_min = min(left_lengths)
    left_max = max(left_lengths)
    
    right_lengths = []
    for pos, seq in zip(anchor_positions, seqs):
        right_len = len(seq) - pos - len(core_parts)
        right_lengths.append(max(0, right_len))
    right_min = min(right_lengths)
    right_max = max(right_lengths)
    
    return (core_parts, left_min, left_max, right_min, right_max)


def merge_rule_and_sequences_v2(covered_seqs):
    """
    多序列锚点比对融合（纪律1：真正的动态方括号）
    
    核心改进：
    - 使用锚点进行多序列比对
    - 对锚点区域逐列比对，生成[AV]等动态简并
    - 锚点两侧用gap表示（交集模式）
    
    修复：
    - 修复1：交集而非并集
    - 修复2：长度差异超过5则拒绝合并
    """
    if len(covered_seqs) < 2:
        return None
    
    # 修复2：长度悬殊检查
    full_seqs = [get_full_sequence(seq) for seq in covered_seqs]
    solid_lens = [len(seq) for seq in full_seqs]
    if max(solid_lens) - min(solid_lens) > MAX_LENGTH_DIFF:
        return None
    
    # 修复1：使用锚点比对（交集模式）
    result = align_multiple_sequences_anchored(full_seqs)
    
    if result is None:
        return None
    
    core_parts, left_min, left_max, right_min, right_max = result
    
    if len(core_parts) < 2:
        return None
    
    # 构建完整规则
    parts = []
    
    # 左侧gap（修复1：只使用交集区域的gap）
    if left_max > 0:
        left_max = min(left_max, MAX_GAP_UPPER)
        left_min = min(left_min, left_max)
        if left_min == left_max:
            parts.append(f'x({left_min})')
        else:
            parts.append(f'x({left_min},{left_max})')
    
    # 核心区（动态简并）
    parts.extend(core_parts)
    
    # 右侧gap（修复1：只使用交集区域的gap）
    if right_max > 0:
        right_max = min(right_max, MAX_GAP_UPPER)
        right_min = min(right_min, right_max)
        if right_min == right_max:
            parts.append(f'x({right_min})')
        else:
            parts.append(f'x({right_min},{right_max})')
    
    return '-'.join(parts)


def build_dynamic_degenerate_core(seqs, start_positions):
    """
    纪律1：构建动态简并核心区（真正的动态方括号）
    
    核心改进：
    - 对每一列进行多序列比对
    - 统计每列实际氨基酸，生成[AV]等
    - 只有>3种氨基酸才拒绝融合
    """
    # 计算最大比对长度
    max_end_pos = max(start_pos + len(seqs[i]) - start_pos for i, start_pos in enumerate(start_positions))
    
    # 从锚点开始位置，逐列比对
    anchor = find_best_anchor_for_multiple(seqs)
    anchor_pos = seqs[0].find(anchor)
    
    # 核心区从锚点开始
    core_start = anchor_pos
    anchor_end = anchor_pos + len(anchor)
    
    # 比对长度：锚点长度 + 前后延伸
    # 但为了简化，我们只处理锚点部分（因为锚点是公共的）
    core_parts = []
    
    # 处理锚点部分
    for col_idx in range(len(anchor)):
        # 找出该列在所有序列中的字符
        chars = []
        for seq_idx, seq in enumerate(seqs):
            pos_in_seq = start_positions[seq_idx] + col_idx
            if pos_in_seq < len(seq):
                char = seq[pos_in_seq]
                if char != '_':
                    chars.append(char)
        
        # 纪律1：生成动态表示
        if len(chars) == 0:
            return None
        
        unique_chars = set(chars)
        
        if len(unique_chars) == 1:
            # 所有序列相同，返回单个氨基酸
            core_parts.append(list(unique_chars)[0])
        elif len(unique_chars) <= MAX_BRACKET_SIZE:
            # 2-3种氨基酸，生成方括号
            bracket = ''.join(sorted(unique_chars))
            core_parts.append(f'[{bracket}]')
        else:
            # >3种氨基酸，拒绝融合
            return None
    
    # 纪律2+3：检查核心区质量
    informative_count = sum(1 for part in core_parts if '[' in part or (len(part) == 1 and part in AA_LETTERS))
    total_count = len(core_parts)
    
    if total_count == 0:
        return None
    
    # 纪律3：有效位占比必须>=70%
    informative_ratio = informative_count / total_count
    if informative_ratio < MIN_INFORMATIVE_RATIO:
        return None
    
    # 纪律2：检查内部连续通配符（核心区不应该有x）
    if 'x' in core_parts or '_' in core_parts:
        return None
    
    return core_parts


def merge_with_additional_sequence_v2(covered_seqs, new_seq):
    """
    将新序列融入现有规则（纪律1：真正的动态方括号）
    
    输入：
    - covered_seqs: 当前已覆盖的序列列表
    - new_seq: 待加入的新序列
    
    输出：更新后的规则（如果合格）
    
    修复：
    - 修复1：交集而非并集
    - 修复2：长度差异超过5则拒绝合并
    """
    # 修复2：长度悬殊检查
    all_seqs = covered_seqs + [new_seq]
    full_seqs = [get_full_sequence(seq) for seq in all_seqs]
    solid_lens = [len(seq) for seq in full_seqs]
    if max(solid_lens) - min(solid_lens) > MAX_LENGTH_DIFF:
        return None
    
    # 修复1：使用锚点比对（交集模式）
    result = align_multiple_sequences_anchored(full_seqs)
    
    if result is None:
        return None
    
    core_parts, left_min, left_max, right_min, right_max = result
    
    if len(core_parts) < 2:
        return None
    
    # 构建完整规则
    parts = []
    
    # 左侧gap
    if left_max > 0:
        left_max = min(left_max, MAX_GAP_UPPER)
        left_min = min(left_min, left_max)
        if left_min == left_max:
            parts.append(f'x({left_min})')
        else:
            parts.append(f'x({left_min},{left_max})')
    
    # 核心区
    parts.extend(core_parts)
    
    # 右侧gap
    if right_max > 0:
        right_max = min(right_max, MAX_GAP_UPPER)
        right_min = min(right_min, right_max)
        if right_min == right_max:
            parts.append(f'x({right_min})')
        else:
            parts.append(f'x({right_min},{right_max})')
    
    return '-'.join(parts)


def find_best_anchor_for_multiple(seqs):
    """
    为多个序列找最佳公共锚点（允许简并）
    
    修改：找到允许每列2-3种氨基酸的最长简并锚点
    返回：(锚点起始位置列表, 简并核心区域)
    """
    if len(seqs) < 2:
        return None
    
    max_len = max(len(seq) for seq in seqs)
    best_score = -1
    best_result = None
    
    # 尝试不同的锚点长度（2-8）
    for anchor_len in range(2, min(max_len, 8) + 1):
        for start_pos in range(max_len - anchor_len + 1):
            chars_list = []
            valid = True
            seq_positions = []
            
            for seq_idx, seq in enumerate(seqs):
                seq_start_pos = -1
                # 在序列中寻找匹配的位置
                for search_start in range(len(seq) - anchor_len + 1):
                    match = True
                    for col_idx in range(anchor_len):
                        pos_in_seq = search_start + col_idx
                        if pos_in_seq >= len(seq):
                            match = False
                            break
                        char = seq[pos_in_seq]
                        if char not in AA_LETTERS:
                            match = False
                            break
                    
                    if match:
                        seq_start_pos = search_start
                        break
                
                if seq_start_pos == -1:
                    valid = False
                    break
                seq_positions.append(seq_start_pos)
            
            if not valid:
                continue
            
            # 检查每列的氨基酸种类
            for col_idx in range(anchor_len):
                chars = []
                for seq_idx, seq in enumerate(seqs):
                    pos_in_seq = seq_positions[seq_idx] + col_idx
                    char = seq[pos_in_seq]
                    if char in AA_LETTERS:
                        chars.append(char)
                
                unique_chars = set(chars)
                if len(unique_chars) > MAX_BRACKET_SIZE:
                    valid = False
                    break
                
                chars_list.append(chars)
            
            if not valid:
                continue
            
            # 计算得分：锚点长度 + 简并程度
            score = anchor_len
            for chars in chars_list:
                if len(set(chars)) > 1:
                    score += 0.5
            
            if score > best_score:
                best_score = score
                # 生成简并核心区域
                core_parts = []
                for chars in chars_list:
                    unique_chars = set(chars)
                    if len(unique_chars) == 1:
                        core_parts.append(list(unique_chars)[0])
                    else:
                        bracket = ''.join(sorted(unique_chars))
                        core_parts.append(f'[{bracket}]')
                
                best_result = (seq_positions, core_parts)
    
    return best_result


def build_dynamic_core(anchor, all_full, anchor_positions):
    """
    纪律1：构建动态简并核心区
    
    核心改进：
    - 不依赖理化家族字典
    - 直接统计每列实际氨基酸，生成[AV]等精准简并
    - 只有>3种氨基酸才降级为x
    """
    # 纪律1：对锚点左右两侧的非公共部分进行动态简并
    # 但由于锚点是公共子串，我们先处理锚点本身
    
    core_parts = []
    
    # 锚点本身直接添加（因为所有序列相同）
    for char in anchor:
        if char == '_':
            core_parts.append('_')
        else:
            core_parts.append(char)
    
    # 纪律2+3：检查核心区质量
    informative_count = sum(1 for part in core_parts if part not in ['x', '_'])
    total_count = len(core_parts)
    
    if total_count == 0:
        return None
    
    # 纪律3：有效位占比必须>=70%
    informative_ratio = informative_count / total_count
    if informative_ratio < MIN_INFORMATIVE_RATIO:
        return None
    
    return core_parts


def build_dynamic_bracket_for_column(seqs, column_idx):
    """
    纪律1：为某一列生成动态方括号
    
    输入：多个序列和列索引
    输出：
    - 如果该列所有序列字符相同：返回单个氨基酸（如'A'）
    - 如果不同但≤3种：返回方括号（如'[AV]'）
    - 如果>3种：返回None（拒绝融合）
    """
    chars_at_column = []
    
    for seq in seqs:
        if column_idx < len(seq):
            char = seq[column_idx]
            if char != '_':  # 下划线作为通配符，不计入统计
                chars_at_column.append(char)
    
    if len(chars_at_column) == 0:
        return '_'  # 该列全是下划线
    
    unique_chars = set(chars_at_column)
    
    # 纪律1：如果只有1种氨基酸，返回单个字符
    if len(unique_chars) == 1:
        return list(unique_chars)[0]
    
    # 纪律1：如果2-3种氨基酸，生成方括号
    if len(unique_chars) <= MAX_BRACKET_SIZE:
        bracket_content = ''.join(sorted(unique_chars))
        return f'[{bracket_content}]'
    
    # 纪律1：如果超过3种氨基酸，拒绝融合
    return None


def find_longest_common_substring(s1, s2, min_len=2):
    """
    找两个序列的最长公共子串
    铁律1：带着下划线一起搜索
    """
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return ''

    # 使用简单的滑动窗口
    max_len = 0
    best_sub = ''

    # 只检查s1中可能的子串（限制长度避免过度匹配）
    for i in range(m):
        for j in range(i + min_len, min(i + 6, m) + 1):
            substr = s1[i:j]
            if substr in s2 and len(substr) > max_len:
                max_len = len(substr)
                best_sub = substr

    return best_sub


def build_rule_from_skeleton(skeleton, seq1, seq2):
    """
    根据骨架构建规则
    铁律4：严格计算gap区间，遍历所有覆盖词计算min和max
    铁律1：保留下划线，正确处理下划线的长度变异
    """
    # 找出骨架在两个序列中的位置
    pos1 = seq1.find(skeleton)
    pos2 = seq2.find(skeleton)

    if pos1 == -1 or pos2 == -1:
        return None

    # 计算gap
    parts = []

    # 铁律4：左侧gap，严格计算min和max
    left1 = pos1
    left2 = pos2
    left_min = min(left1, left2)
    left_max = max(left1, left2)

    if left_max > 0:
        # 铁律3：限制gap上限
        left_max = min(left_max, MAX_GAP_UPPER)
        left_min = min(left_min, left_max)
        if left_min == left_max:
            parts.append(f'x({left_min})')
        else:
            parts.append(f'x({left_min},{left_max})')

    # 骨架本身
    # 铁律1：保留下划线，如果骨架中包含下划线，保持原样
    for char in skeleton:
        if char == '_':
            # 下划线作为通配符
            parts.append('_')
        else:
            parts.append(char)

    # 铁律4：右侧gap，严格计算min和max
    right1 = len(seq1) - pos1 - len(skeleton)
    right2 = len(seq2) - pos2 - len(skeleton)
    right_min = min(right1, right2)
    right_max = max(right1, right2)

    if right_max > 0:
        right_max = min(right_max, MAX_GAP_UPPER)
        right_min = min(right_min, right_max)
        if right_min == right_max:
            parts.append(f'x({right_min})')
        else:
            parts.append(f'x({right_min},{right_max})')

    return '-'.join(parts)


# ============================================================
# 模块4：主循环（铁律3：微簇规模控制）
# ============================================================

def mine_rules(pos_set, background_pool):
    """
    序贯覆盖算法挖掘规则（纪律1+2+3）
    
    纪律1：动态局部简并
    纪律2：严禁内部连续通配符
    纪律3：最低有效位占比>=70%
    """
    uncovered_pool = sort_sequences_by_centrality(list(pos_set))
    final_rules = []
    
    # K-mer反向索引
    kmer_index = _build_kmer_index(uncovered_pool)
    
    iteration = 0
    max_iterations = 500
    patience_limit = 20
    
    while uncovered_pool and iteration < max_iterations:
        iteration += 1
        
        # 取出得分最高的种子
        seed_seq = uncovered_pool.pop(0)
        
        # 纪律1：初始化，使用多序列融合逻辑
        covered_seqs = [seed_seq]
        
        # K-mer预过滤
        seed_full = get_full_sequence(seed_seq)
        seed_kmers = _extract_kmers(seed_full)
        candidate_pool = _filter_by_kmers(uncovered_pool, seed_kmers, kmer_index)
        
        if not candidate_pool:
            continue
        
        remaining_pool = []
        consecutive_fails = 0
        
        # 尝试添加更多序列（最多3个，总共4个）
        for candidate_seq in uncovered_pool:
            if candidate_seq not in candidate_pool:
                remaining_pool.append(candidate_seq)
                continue
            
            # 达到最大覆盖数（4个），停止
            if len(covered_seqs) >= 4:
                remaining_pool.append(candidate_seq)
                continue
            
            # 纪律1：尝试融合新序列（使用v2版本）
            temp_rule = merge_with_additional_sequence_v2(covered_seqs, candidate_seq)
            
            if temp_rule is None:
                remaining_pool.append(candidate_seq)
                consecutive_fails += 1
                if consecutive_fails >= patience_limit:
                    break
                continue
            
            # 纪律2+3：校验规则（严禁连续x + 有效位占比>=70%）
            if is_rule_valid(temp_rule, background_pool):
                covered_seqs.append(candidate_seq)
                consecutive_fails = 0
            else:
                remaining_pool.append(candidate_seq)
                consecutive_fails += 1
                if consecutive_fails >= patience_limit:
                    for rest in uncovered_pool[uncovered_pool.index(candidate_seq)+1:]:
                        if rest not in remaining_pool:
                            remaining_pool.append(rest)
                    break
        
        # 结算判断，只接受2-4个词的微簇
        if len(covered_seqs) >= 2 and len(covered_seqs) <= 4:
            # 纪律1：生成最终规则（使用v2版本，真正的动态方括号）
            final_rule = merge_rule_and_sequences_v2(covered_seqs)
            
            if final_rule:
                # 修复3：回溯自证——验证规则真的能匹配所有声称覆盖的词
                if not validate_rule_coverage(final_rule, covered_seqs):
                    continue
                
                # 原有规则有效性检查
                if is_rule_valid(final_rule, background_pool):
                    final_rules.append((final_rule, covered_seqs))
        
        uncovered_pool = remaining_pool
        
        if iteration % 10 == 0:
            print(f"    迭代 {iteration}: 剩余 {len(uncovered_pool)} 个词")
    
    return final_rules


def validate_rule_coverage(rule_pattern, covered_seqs):
    """
    修复3：回溯自证（Retroactive Validation）
    
    将规则编译为正则，使用fullmatch验证每一个声称覆盖的词
    只要有任何一个词匹配失败，规则作废
    
    返回：True如果所有词都能匹配，False否则
    """
    regex_str = rule_to_regex(rule_pattern)
    
    try:
        compiled = re.compile(regex_str)
    except re.error:
        return False
    
    for seq in covered_seqs:
        full_seq = get_full_sequence(seq)
        solid_seq = get_solid_sequence(full_seq)
        
        # 必须完全匹配（fullmatch）
        if not compiled.fullmatch(solid_seq):
            return False
    
    return True


def _build_kmer_index(seq_list, k=3):
    """
    构建K-mer反向索引
    铁律1：保留下划线
    """
    index = defaultdict(set)
    for seq in seq_list:
        # 铁律1：使用完整序列
        full_seq = get_full_sequence(seq)
        solid = get_solid_sequence(full_seq)
        for i in range(len(solid) - k + 1):
            kmer = solid[i:i+k]
            index[kmer].add(seq)
    return index


def _extract_kmers(seq, k=3):
    """提取序列中的所有K-mer（实心氨基酸）"""
    kmers = set()
    solid = get_solid_sequence(seq)
    for i in range(len(solid) - k + 1):
        kmers.add(solid[i:i+k])
    return kmers


def _filter_by_kmers(pool, seed_kmers, kmer_index):
    """根据K-mer预过滤候选序列"""
    candidates = set()
    for kmer in seed_kmers:
        if kmer in kmer_index:
            candidates.update(kmer_index[kmer])
    return candidates


# ============================================================
# 主Pipeline
# ============================================================

def process_fragment(frag_id, P, global_words, wordfrag2score):
    """处理单个分子片段"""
    B = global_words - P

    # 挖掘规则
    rules_data = mine_rules(P, B)

    # 转换为输出格式
    output_rules = []
    for rule_pattern, covered_words in rules_data:
        scores = []
        for word in covered_words:
            key = f'{word} {frag_id}'
            if key in wordfrag2score:
                scores.append(float(wordfrag2score[key]))

        if scores:
            scores_sorted = sorted(scores)
            trim = max(1, len(scores_sorted) // 10)
            if len(scores_sorted) > 2 * trim:
                trimmed = scores_sorted[trim:-trim]
            else:
                trimmed = scores_sorted
            avg_score = sum(trimmed) / len(trimmed) if trimmed else 0.0
        else:
            avg_score = 0.0

        # 铁律2：计算FPR（使用fullmatch）
        regex_str = rule_to_regex(rule_pattern)
        compiled = re.compile(regex_str)

        bg_list = list(B)
        if len(bg_list) > FPR_SAMPLE_SIZE:
            bg_list = random.sample(bg_list, FPR_SAMPLE_SIZE)

        bg_count = 0
        max_fp = int(MAX_FPR * len(bg_list))
        for word in bg_list:
            # 铁律1：使用完整序列
            full_seq = get_full_sequence(word)
            # 铁律2：使用fullmatch
            if compiled.fullmatch(full_seq):
                bg_count += 1
                if bg_count > max_fp:
                    break

        fpr = bg_count / len(bg_list) if bg_list else 0.0

        output_rules.append({
            'Pattern': rule_pattern,
            'Average_Score': round(avg_score, 4),
            'Covered_Count': len(covered_words),
            'Covered_Words': list(covered_words),
            'False_Positive_Rate': round(fpr, 4),
        })

    # 按Covered_Count降序排序
    output_rules.sort(key=lambda x: x['Covered_Count'], reverse=True)

    compression_ratio = len(P) / len(output_rules) if output_rules else 0

    return {
        'Fragment_ID': frag_id,
        'Total_Original_Words': len(P),
        'Compression_Ratio': round(compression_ratio, 2),
        'Rules': output_rules,
    }


def main():
    parser = argparse.ArgumentParser(description='蛋白词规则挖掘 (v12.0)')
    parser.add_argument('--first', type=int, default=10,
                        help='只处理前N个fragment (默认10)')
    parser.add_argument('--output', type=str, default='rule_results.json',
                        help='输出文件路径')
    args = parser.parse_args()

    print("=" * 60)
    print("加载数据...")
    t0 = time.time()

    frag2words, word2frags, wordfrag2score, global_words = load_data()

    # 铁律1：预计算氨基酸缓存（保留下划线）
    set_global_aa_seqs(global_words)

    print(f"  全局词库: {len(global_words)} 个蛋白词")
    print(f"  分子片段数: {len(frag2words)}")
    print(f"  加载耗时: {time.time() - t0:.1f}s")

    # 选取前N个fragment
    frag_ids = sorted(frag2words.keys(), key=lambda x: int(x.split('_')[1]))
    selected_frags = frag_ids[:args.first]

    print(f"\n处理前 {args.first} 个分子片段...")
    print("=" * 60)

    results = []
    total_start = time.time()

    for i, frag_id in enumerate(selected_frags):
        frag_start = time.time()
        P = frag2words[frag_id]
        print(f"\n[{i+1}/{args.first}] 处理 {frag_id} (正样本数: {len(P)})...")

        result = process_fragment(frag_id, P, global_words, wordfrag2score)

        frag_time = time.time() - frag_start
        result['Process_Time_s'] = round(frag_time, 2)
        results.append(result)

        print(f"  规则数: {len(result['Rules'])}, "
              f"压缩率: {result['Compression_Ratio']}, "
              f"耗时: {frag_time:.1f}s")
        for rule in result['Rules']:
            print(f"    {rule['Pattern']} "
                  f"(覆盖:{rule['Covered_Count']}, "
                  f"FPR:{rule['False_Positive_Rate']}, "
                  f"Score:{rule['Average_Score']})")

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"总耗时: {total_time:.1f}s")
    print(f"结果已保存到: {args.output}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == '__main__':
    main()