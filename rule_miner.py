#!/usr/bin/env python3
"""
蛋白词-分子片段规则挖掘算法
基于信息熵与 Set Cover 的全局寻优

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

import numpy as np

# ============================================================
# 常量定义
# ============================================================

AA_LETTERS = set('ACDEFGHIKLMNPQRSTVWY')

AMINO_ACID_GROUPS = {
    '[DE]': {'D', 'E'},
    '[KRH]': {'K', 'R', 'H'},
    '[FYW]': {'F', 'Y', 'W'},
    '[LIVM]': {'L', 'I', 'V', 'M'},
    '[ST]': {'S', 'T'},
    '[NQ]': {'N', 'Q'},
    '[GAS]': {'G', 'A', 'S'},
}

# 语义化氨基酸族名（PROSITE风格）
# 若某列的氨基酸集合是某族的子集，则用语义族名表示
SEMANTIC_GROUPS = [
    ('acidic',   {'D', 'E'}),
    ('basic',    {'K', 'R', 'H'}),
    ('aromatic', {'F', 'Y', 'W'}),
    ('aliphatic',{'L', 'I', 'V', 'M'}),
    ('hydroxyl', {'S', 'T'}),
    ('amide',    {'N', 'Q'}),
    ('small',    {'G', 'A', 'S'}),
]

# 语义族名 → 氨基酸正则（用于正则编译）
SEMANTIC_TO_REGEX = {name: '[' + ''.join(sorted(aas)) + ']' for name, aas in SEMANTIC_GROUPS}

MAX_IC = math.log2(20)  # ~4.32 bits
MIN_IC_SCORE = 1.5       # 修正：从3.0下调至1.5
MIN_IC_STRONG = 3.0       # 强保守阈值（用于优先简并）
TOP_N_SEEDS = 100         # 修正：用于模糊聚类的种子池大小
TOP_N_SEEDS_FINAL = 20    # 模糊聚类后最终输出的种子数
MIN_KMER_LEN = 3
MAX_KMER_LEN = 8
SLIDING_WINDOW = 4
MAX_FPR = 0.01
EPSILON = 1e-10
HAMMING_DIST = 1          # 模糊聚类：仅允许1个位置不同
FREQ_THRESHOLD = 0.8      # 弱保守列：取累积频率超80%的Top-K氨基酸
TERMINAL_GAP_MAX_RANGE = 3  # 头尾GAP允许的最大范围(max-min)，超过则不加GAP

# Set Cover权重
W1 = 0.5
W2 = 0.3
W3 = 0.2


# ============================================================
# 数据加载
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
# 工具函数
# ============================================================

def aa_sequence(word):
    """提取氨基酸序列（去掉 '_'）"""
    return ''.join(c for c in word if c in AA_LETTERS)


def extract_kmers_from_words(words, min_len=MIN_KMER_LEN, max_len=MAX_KMER_LEN):
    """从一组蛋白词中提取所有k-mer"""
    kmer_to_words = defaultdict(set)
    for word in words:
        aa_seq = aa_sequence(word)
        n = len(aa_seq)
        for length in range(min_len, min(max_len, n) + 1):
            for i in range(n - length + 1):
                kmer = aa_seq[i:i + length]
                kmer_to_words[kmer].add(word)
    return dict(kmer_to_words)


def compute_ic_score(aa_counts):
    """计算一列的IC分数"""
    total = sum(aa_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in aa_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return MAX_IC - entropy


def _aa_set_to_block(aa_set):
    """
    将氨基酸集合转换为语义化表示
    - 单氨基酸 → 直接返回字母
    - 全属于同一语义族 → 返回语义族名，如 [acidic]
    - 跨族 → 返回原始方括号
    """
    if len(aa_set) == 1:
        return list(aa_set)[0]

    # 优先匹配语义族：若集合是某个族的子集，用语义名
    for name, group_set in SEMANTIC_GROUPS:
        if aa_set.issubset(group_set):
            return f'[{name}]'

    # 回退到原始方括号
    sorted_aas = ''.join(sorted(aa_set))
    return f'[{sorted_aas}]'


def get_block_representation(aa_counts, ic=None):
    """
    根据氨基酸分布和IC分数返回最优简并表示（PROSITE语义化）

    三级判定：
    - IC >= 3.0: 强保守，单氨基酸或语义族简并
    - 1.5 <= IC < 3.0: 弱保守，取累积频率超80%的Top-K氨基酸
    - IC < 1.5: 极度柔性，返回 None（熔断）
    """
    total = sum(aa_counts.values())
    if total == 0:
        return 'x'

    if ic is None:
        ic = compute_ic_score(aa_counts)

    # 极度柔性区 → 熔断
    if ic < MIN_IC_SCORE:
        return None

    unique_aas = set(aa_counts.keys())

    # 强保守：单氨基酸或语义族简并
    if ic >= MIN_IC_STRONG:
        return _aa_set_to_block(unique_aas)

    # 弱保守 (1.5 <= IC < 3.0): 取累积频率超80%的Top-K氨基酸
    sorted_items = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)
    cumsum = 0
    selected = []
    for aa, count in sorted_items:
        selected.append(aa)
        cumsum += count
        if cumsum / total >= FREQ_THRESHOLD:
            break

    return _aa_set_to_block(set(selected))


def block_to_regex(block):
    """将一个block转换为正则表达式片段，支持语义族名"""
    if block.startswith('[') and block.endswith(']'):
        inner = block[1:-1]
        # 语义族名 → 展开为氨基酸正则
        if inner in SEMANTIC_TO_REGEX:
            return SEMANTIC_TO_REGEX[inner]
        return block
    if block == 'x':
        return '.'
    if len(block) == 1 and block in AA_LETTERS:
        return block
    return block


def rule_to_regex(rule_pattern):
    """将规则模式转换为正则表达式，支持 x(m,n) PROSITE格式"""
    parts = rule_pattern.split('-')
    regex_parts = []
    for part in parts:
        if part.startswith('x(') and part.endswith(')'):
            # PROSITE格式: x(m,n) 或 x(n)
            inner = part[2:-1]
            if ',' in inner:
                m, n = inner.split(',')
            else:
                m = n = inner
            regex_parts.append(f'.{{{m},{n}}}')
        elif part.startswith('<GAP(') and part.endswith(')>'):
            # 兼容旧格式
            inner = part[5:-2]
            m, n = inner.split(',')
            regex_parts.append(f'.{{{m},{n}}}')
        else:
            regex_parts.append(block_to_regex(part))
    return ''.join(regex_parts)


def count_matches(rule_pattern, words):
    """统计规则在词集合中匹配的数量和匹配词"""
    regex_str = rule_to_regex(rule_pattern)
    compiled = re.compile(regex_str)
    matched = set()
    for word in words:
        aa_seq = aa_sequence(word)
        if compiled.search(aa_seq):
            matched.add(word)
    return len(matched), matched


# ============================================================
# Hamming 距离计算
# ============================================================

def hamming_distance(s1, s2):
    """计算两个等长字符串的汉明距离"""
    if len(s1) != len(s2):
        return float('inf')
    return sum(1 for a, b in zip(s1, s2) if a != b)


def merge_kmers(kmer1, kmer2):
    """
    合并两个仅差1个位置的k-mer为模糊种子
    如 'YRG' 和 'FRG' → '[FY]-R-G'
    """
    if len(kmer1) != len(kmer2):
        return None
    parts = []
    for a, b in zip(kmer1, kmer2):
        if a == b:
            parts.append(a)
        else:
            parts.append('[' + ''.join(sorted([a, b])) + ']')
    return '-'.join(parts)


# ============================================================
# Phase 1: 高判别力锚点挖掘 + 种子模糊化
# ============================================================

def phase1_anchor_extraction(P, global_kmer_freq, global_total):
    """
    从正样本中提取高判别力锚点，并进行模糊聚类

    1. 提取Top-100 k-mers
    2. 对同长度k-mers计算汉明距离，合并差1个位置的种子
    3. 重新计算合并后的AnchorScore
    4. 取Top-20输出

    返回: [(seed_pattern, anchor_score, matching_words), ...]
    """
    pos_kmer_to_words = extract_kmers_from_words(P)

    # 计算原始AnchorScore
    scores = []
    for kmer, matching_words in pos_kmer_to_words.items():
        support_p = len(matching_words)
        freq_u = global_kmer_freq.get(kmer, 0) / global_total
        if freq_u == 0:
            freq_u = EPSILON
        anchor_score = math.log2(support_p + 1) * (-math.log2(freq_u))
        scores.append((kmer, anchor_score, matching_words))

    # 按score降序排列，取Top-100
    scores.sort(key=lambda x: x[1], reverse=True)
    top_100 = scores[:TOP_N_SEEDS]

    # ---- 种子模糊聚类 ----
    # 按长度分组
    by_length = defaultdict(list)
    for kmer, score, words in top_100:
        by_length[len(kmer)].append((kmer, score, words))

    merged = {}  # seed_pattern -> (merged_score, merged_words)

    for length, items in by_length.items():
        used = [False] * len(items)

        for i in range(len(items)):
            if used[i]:
                continue
            kmer_i, score_i, words_i = items[i]
            best_j = -1
            best_score_j = -1

            # 找汉明距离=1的最近邻
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                kmer_j, score_j, words_j = items[j]
                if hamming_distance(kmer_i, kmer_j) == HAMMING_DIST:
                    if score_j > best_score_j:
                        best_score_j = score_j
                        best_j = j

            if best_j != -1:
                # 合并
                kmer_j, score_j, words_j = items[best_j]
                merged_pattern = merge_kmers(kmer_i, kmer_j)
                merged_words = words_i | words_j
                merged_support = len(merged_words)
                freq_u = global_kmer_freq.get(kmer_i, 0) / global_total  # 用原始kmer的频率近似
                if freq_u == 0:
                    freq_u = EPSILON
                merged_score = math.log2(merged_support + 1) * (-math.log2(freq_u))
                merged[merged_pattern] = (merged_score, merged_words)
                used[i] = True
                used[best_j] = True
            else:
                # 无需合并，直接保留原始种子作为"模糊种子"（纯字母）
                pattern = '-'.join(list(kmer_i))
                merged[pattern] = (score_i, words_i)
                used[i] = True

    # 按score排序，取Top-20
    merged_scores = [(pat, sc, wds) for pat, (sc, wds) in merged.items()]
    merged_scores.sort(key=lambda x: x[1], reverse=True)

    return merged_scores[:TOP_N_SEEDS_FINAL]


# ============================================================
# Phase 2: 基于滑动窗口的局部块延展
# ============================================================

def get_aligned_columns(words, seed):
    """
    以seed对齐所有包含seed的正样本词

    seed 可能是模糊种子，如 '[YF]-R-G' 形式（用'-'分隔各列）
    或普通种子如 'YRG'（纯字母）

    返回: {
        'right_columns': {offset: Counter},
        'left_columns': {offset: Counter},
        'seed_columns': [Counter, ...],
        'aligned_words': set,
        'seed_positions': {word: idx},  # seed在aa_seq中的起始位置
        'aa_lengths': {word: len},       # 每个对齐词的aa序列长度
    }
    """
    # 解析seed为列列表
    seed_cols = seed.split('-') if '-' in seed else list(seed)
    seed_len = len(seed_cols)

    # 为每列编译正则
    col_regexes = []
    for col in seed_cols:
        if col.startswith('[') and col.endswith(']'):
            col_regexes.append(re.compile(col))
        else:
            col_regexes.append(re.compile(re.escape(col)))

    right_columns = defaultdict(Counter)
    left_columns = defaultdict(Counter)
    seed_columns = [Counter() for _ in range(seed_len)]
    aligned_words = set()
    seed_positions = {}
    aa_lengths = {}

    for word in words:
        aa_seq = aa_sequence(word)
        n = len(aa_seq)
        aa_lengths[word] = n

        # 在aa_seq中查找匹配seed的起始位置
        idx = _find_seed_regex(aa_seq, col_regexes)
        if idx == -1:
            continue

        aligned_words.add(word)
        seed_positions[word] = idx

        # seed各列
        for i in range(seed_len):
            seed_columns[i][aa_seq[idx + i]] += 1

        # 左边
        for offset in range(-1, -idx - 1, -1):
            pos = idx + offset
            if pos >= 0:
                left_columns[offset][aa_seq[pos]] += 1

        # 右边
        right_limit = n - (idx + seed_len)
        for offset in range(1, right_limit + 1):
            pos = idx + seed_len + offset - 1
            if pos < n:
                right_columns[offset][aa_seq[pos]] += 1

    return {
        'right_columns': dict(right_columns),
        'left_columns': dict(left_columns),
        'seed_columns': seed_columns,
        'aligned_words': aligned_words,
        'seed_positions': seed_positions,
        'aa_lengths': aa_lengths,
        'seed_len': seed_len,
    }


def _find_seed_regex(aa_seq, col_regexes):
    """在氨基酸序列中查找匹配模糊种子的位置"""
    n = len(aa_seq)
    seed_len = len(col_regexes)
    for i in range(n - seed_len + 1):
        match = True
        for j, regex in enumerate(col_regexes):
            if not regex.match(aa_seq[i + j]):
                match = False
                break
        if match:
            return i
    return -1


def _expand_direction(columns, seed_part, words, B, P, direction='right'):
    """
    向一侧延展，在滑动窗口内找IC最高的位置

    修正：
    - IC >= 3.0: 强保守块，优先简并
    - 1.5 <= IC < 3.0: 弱保守块，取80%频率的Top-K氨基酸
    - IC < 1.5: 熔断

    返回: (candidates, min_offset_used, max_offset_used)
    candidates: [(rule_pattern, matched_words, total_ic, fpr), ...]
    """
    candidates = []
    current_block = seed_part

    if direction == 'right':
        current_end = 0
    else:
        current_end = 0

    used_offsets = set()
    min_offset = 0  # 该方向使用的最小offset（最左）
    max_offset = 0  # 该方向使用的最大offset（最右）

    while True:
        if direction == 'right':
            candidate_offsets = sorted([k for k in columns.keys()
                                        if k > current_end and k not in used_offsets])
        else:
            candidate_offsets = sorted([k for k in columns.keys()
                                        if k < current_end and k not in used_offsets],
                                       reverse=True)

        if not candidate_offsets:
            break

        best_offset = None
        best_ic = -1
        best_block = None

        for offset in candidate_offsets:
            dist = abs(offset - current_end)
            if dist > SLIDING_WINDOW:
                break

            aa_counts = columns[offset]
            ic = compute_ic_score(aa_counts)
            if ic >= MIN_IC_SCORE and ic > best_ic:
                best_ic = ic
                best_offset = offset
                best_block = get_block_representation(aa_counts, ic)

        if best_offset is None:
            break

        used_offsets.add(best_offset)
        min_offset = min(min_offset, best_offset)
        max_offset = max(max_offset, best_offset)

        # 计算gap
        if direction == 'right':
            gap = best_offset - current_end - 1
        else:
            gap = current_end - best_offset - 1
        gap = max(0, gap)

        # 构建规则（PROSITE格式: x(m,n)）
        if gap == 0:
            if direction == 'right':
                new_rule = f'{current_block}-{best_block}'
            else:
                new_rule = f'{best_block}-{current_block}'
        else:
            if direction == 'right':
                new_rule = f'{current_block}-x({gap},{gap + 2})-{best_block}'
            else:
                new_rule = f'{best_block}-x({gap},{gap + 2})-{current_block}'

        # 检查FPR
        fpr = 0.0
        if B:
            mc, _ = count_matches(new_rule, B)
            fpr = mc / len(B)

        if fpr > MAX_FPR:
            break

        _, matched = count_matches(new_rule, P)

        candidates.append((new_rule, matched, best_ic, fpr))

        current_block = new_rule
        current_end = best_offset

    return candidates, min_offset, max_offset


def _add_terminal_gaps(rule_pattern, aligned, left_offsets_used, right_offsets_used):
    """
    头尾闭环处理：在规则首尾添加 x(m,n) 以覆盖完整序列长度

    收紧条件：
    - 仅当 max_overhang > 0 且 max - min <= TERMINAL_GAP_MAX_RANGE 时才添加
    - 避免添加过宽的GAP（如 x(0,17)），否则规则太泛化

    返回: 带首尾GAP的规则字符串（PROSITE格式）
    """
    seed_len = aligned['seed_len']
    seed_positions = aligned['seed_positions']
    aa_lengths = aligned['aa_lengths']

    if left_offsets_used:
        leftmost_offset = min(left_offsets_used)
    else:
        leftmost_offset = 0

    if right_offsets_used:
        rightmost_offset = max(right_offsets_used)
    else:
        rightmost_offset = 0

    left_overhangs = []
    right_overhangs = []

    for word in aligned['aligned_words']:
        idx = seed_positions[word]
        seq_len = aa_lengths[word]

        if leftmost_offset < 0:
            left_pos = idx + leftmost_offset
        else:
            left_pos = idx
        left_overhang = left_pos
        left_overhangs.append(left_overhang)

        if rightmost_offset > 0:
            right_pos = idx + seed_len + rightmost_offset - 1
        else:
            right_pos = idx + seed_len - 1
        right_overhang = seq_len - right_pos - 1
        right_overhangs.append(right_overhang)

    if not left_overhangs or not right_overhangs:
        return rule_pattern

    min_left = min(left_overhangs)
    max_left = max(left_overhangs)
    min_right = min(right_overhangs)
    max_right = max(right_overhangs)

    # 拼接规则：仅在GAP范围紧凑时添加
    parts = []
    if max_left > 0 and (max_left - min_left) <= TERMINAL_GAP_MAX_RANGE:
        parts.append(f'x({min_left},{max_left})')
    parts.append(rule_pattern)
    if max_right > 0 and (max_right - min_right) <= TERMINAL_GAP_MAX_RANGE:
        parts.append(f'x({min_right},{max_right})')

    return '-'.join(parts)


def phase2_expand_from_seed(seed_pattern, seed_matching_words, P, B, seed_score):
    """
    从模糊种子出发，双向延展生成候选规则，并进行头尾闭环

    返回: [(rule_pattern, matched_words, ic_sum, fpr), ...]
    """
    aligned = get_aligned_columns(seed_matching_words, seed_pattern)

    # 计算seed本身的IC
    seed_ic_sum = 0.0
    seed_blocks = []
    for col in aligned['seed_columns']:
        ic = compute_ic_score(col)
        seed_ic_sum += ic
        block = get_block_representation(col, ic)
        if block is None:
            block = get_block_representation(col, compute_ic_score(col))  # fallback
        seed_blocks.append(block)

    seed_rule = '-'.join(seed_blocks)

    # 检查seed本身作为规则的FPR
    fpr = _calc_fpr(seed_rule, B)
    candidates = []

    if fpr <= MAX_FPR:
        _, matched = count_matches(seed_rule, P)
        # 头尾闭环
        closed_rule = _add_terminal_gaps(seed_rule, aligned, set(), set())
        _, closed_matched = count_matches(closed_rule, P)
        closed_fpr = _calc_fpr(closed_rule, B)
        candidates.append((closed_rule, closed_matched, seed_ic_sum, closed_fpr))

    # 向右延展
    right_candidates, _, right_max = _expand_direction(
        aligned['right_columns'], seed_rule, seed_matching_words, B, P, 'right'
    )
    right_offsets = {right_max} if right_max != 0 else set()

    for rp, mw, ic, fpr in right_candidates:
        closed_rule = _add_terminal_gaps(rp, aligned, set(), right_offsets)
        _, closed_matched = count_matches(closed_rule, P)
        closed_fpr = _calc_fpr(closed_rule, B)
        if closed_fpr <= MAX_FPR:
            candidates.append((closed_rule, closed_matched, ic, closed_fpr))

    # 向左延展
    left_candidates, left_min, _ = _expand_direction(
        aligned['left_columns'], seed_rule, seed_matching_words, B, P, 'left'
    )
    left_offsets = {left_min} if left_min != 0 else set()

    for lp, lw, ic, fpr in left_candidates:
        closed_rule = _add_terminal_gaps(lp, aligned, left_offsets, set())
        _, closed_matched = count_matches(closed_rule, P)
        closed_fpr = _calc_fpr(closed_rule, B)
        if closed_fpr <= MAX_FPR:
            candidates.append((closed_rule, closed_matched, ic, closed_fpr))

    return candidates


def _calc_fpr(rule_pattern, B):
    if len(B) == 0:
        return 0.0
    mc, _ = count_matches(rule_pattern, B)
    return mc / len(B)


# ============================================================
# Phase 3: 集合覆盖全局寻优
# ============================================================

def phase3_set_cover(candidates, P):
    """从候选规则中通过贪心集合覆盖选择最优规则组合"""
    if not candidates:
        return []

    uncovered = set(P)
    remaining = [(rp, set(mw), ic, fpr) for rp, mw, ic, fpr in candidates]

    if not remaining:
        return []

    max_ic = max(r[2] for r in remaining)
    min_ic = min(r[2] for r in remaining)
    max_fpr = max(r[3] for r in remaining)
    min_fpr = min(r[3] for r in remaining)

    selected = []

    while uncovered and remaining:
        best_utility = -float('inf')
        best_idx = -1
        best_rule = None
        best_covered = None

        for i, (rp, mw, ic, fpr) in enumerate(remaining):
            delta = mw & uncovered
            if len(delta) == 0:
                continue

            delta_norm = len(delta) / len(P)
            ic_range = max_ic - min_ic if max_ic != min_ic else 1.0
            ic_norm = (ic - min_ic) / ic_range
            fpr_range = max_fpr - min_fpr if max_fpr != min_fpr else 1.0
            fpr_norm = (fpr - min_fpr) / fpr_range

            utility = W1 * delta_norm + W2 * ic_norm - W3 * fpr_norm

            if utility > best_utility:
                best_utility = utility
                best_idx = i
                best_rule = (rp, ic, fpr)
                best_covered = delta

        if best_idx == -1:
            break

        selected.append((best_rule[0], best_covered, best_rule[1], best_rule[2]))
        uncovered -= best_covered
        remaining.pop(best_idx)

    return selected


# ============================================================
# Phase 4: 置信度聚合与输出
# ============================================================

def phase4_aggregate(selected_rules, wordfrag2score, frag_id):
    """聚合置信度打分"""
    output_rules = []
    for rule_pattern, covered_words, ic_sum, fpr in selected_rules:
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

        output_rules.append({
            'Pattern': rule_pattern,
            'Average_Score': round(avg_score, 4),
            'Covered_Count': len(covered_words),
            'Rule_IC_Score': round(ic_sum, 2),
            'False_Positive_Rate': round(fpr, 4),
        })

    return output_rules


# ============================================================
# 主Pipeline
# ============================================================

def process_fragment(frag_id, P, global_words, global_kmer_freq, global_total,
                     wordfrag2score):
    """处理单个分子片段"""
    B = global_words - P

    # Phase 1: 锚点挖掘（含模糊聚类）
    seeds = phase1_anchor_extraction(P, global_kmer_freq, global_total)

    # Phase 2: 对每个seed做双向延展 + 头尾闭环
    all_candidates = []
    for seed_pattern, seed_score, seed_words in seeds:
        candidates = phase2_expand_from_seed(seed_pattern, seed_words, P, B, seed_score)
        all_candidates.extend(candidates)

    # 去重候选规则
    unique_candidates = {}
    for rp, mw, ic, fpr in all_candidates:
        if rp not in unique_candidates:
            unique_candidates[rp] = (rp, mw, ic, fpr)
        elif len(mw) > len(unique_candidates[rp][1]):
            unique_candidates[rp] = (rp, mw, ic, fpr)

    all_candidates = list(unique_candidates.values())

    # Phase 3: 集合覆盖
    selected = phase3_set_cover(all_candidates, P)

    # Phase 4: 聚合输出
    rules = phase4_aggregate(selected, wordfrag2score, frag_id)

    compression_ratio = len(P) / len(rules) if rules else 0

    return {
        'Fragment_ID': frag_id,
        'Total_Original_Words': len(P),
        'Compression_Ratio': round(compression_ratio, 2),
        'Rules': rules,
    }


def main():
    parser = argparse.ArgumentParser(description='蛋白词规则挖掘')
    parser.add_argument('--first', type=int, default=10,
                        help='只处理前N个fragment (默认10)')
    parser.add_argument('--output', type=str, default='rule_results.json',
                        help='输出文件路径')
    args = parser.parse_args()

    print("=" * 60)
    print("加载数据...")
    t0 = time.time()

    frag2words, word2frags, wordfrag2score, global_words = load_data()

    print(f"  全局词库: {len(global_words)} 个蛋白词")
    print(f"  分子片段数: {len(frag2words)}")
    print(f"  加载耗时: {time.time() - t0:.1f}s")

    # 预计算全局k-mer频率
    print("\n预计算全局k-mer频率...")
    t1 = time.time()
    global_kmer_freq = {}
    global_kmer_to_words = extract_kmers_from_words(global_words)
    for kmer, words in global_kmer_to_words.items():
        global_kmer_freq[kmer] = len(words)
    global_total = len(global_words)
    print(f"  全局k-mer数: {len(global_kmer_freq)}")
    print(f"  预计算耗时: {time.time() - t1:.1f}s")

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

        result = process_fragment(
            frag_id, P, global_words, global_kmer_freq, global_total,
            wordfrag2score
        )

        frag_time = time.time() - frag_start
        result['Process_Time_s'] = round(frag_time, 2)
        results.append(result)

        # 打印摘要
        print(f"  规则数: {len(result['Rules'])}, "
              f"压缩率: {result['Compression_Ratio']}, "
              f"耗时: {frag_time:.1f}s")
        for rule in result['Rules']:
            print(f"    {rule['Pattern']} "
                  f"(覆盖:{rule['Covered_Count']}, "
                  f"IC:{rule['Rule_IC_Score']}, "
                  f"FPR:{rule['False_Positive_Rate']}, "
                  f"Score:{rule['Average_Score']})")

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"总耗时: {total_time:.1f}s")
    print(f"结果已保存到: {args.output}")

    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == '__main__':
    main()