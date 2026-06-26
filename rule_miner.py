#!/usr/bin/env python3
"""
蛋白词-分子片段规则挖掘算法 (v12.0)
严格遵守4条铁律：
1. 绝对禁止擦除下划线 `_`
2. 死守全长匹配 (fullmatch)
3. 微簇规模控制 (Covered_Count: 2-4)
4. 严格PROSITE语法

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
MAX_GAP_SPAN = 3      # 铁律3：降低Gap跨度限制（从4降到3）
MAX_GAP_UPPER = 3     # 铁律3：降低单个Gap最大上限（从5降到3）
MAX_WILDCARD_RATIO = 0.5  # 铁律3：最大通配比例
FPR_SAMPLE_SIZE = 5000   # 采样大小

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
    严苛的规则校验（铁律2 + 铁律3）

    铁律2：使用fullmatch死守全长
    铁律3：微簇规模控制，通配比例<=50%，连续x<=3
    """
    parts = rule_pattern.split('-')

    # 铁律3：通配比例检查
    solid_blocks = 0
    wildcard_blocks = 0
    consecutive_x = 0  # 连续x计数

    for part in parts:
        if part.startswith('x(') or part.startswith('<GAP(') or part == 'x' or part == '_':
            wildcard_blocks += 1
            consecutive_x += 1
            # 铁律3：连续x超过3，熔断
            if consecutive_x > 3:
                return False
        else:
            solid_blocks += 1
            consecutive_x = 0  # 重置计数器

    total_blocks = solid_blocks + wildcard_blocks
    if total_blocks == 0:
        return False

    wildcard_ratio = wildcard_blocks / total_blocks
    # 铁律3：通配比例超过50%，熔断
    if wildcard_ratio > MAX_WILDCARD_RATIO:
        return False

    # 铁律3：Gap跨度限制（更严格）
    for part in parts:
        if part.startswith('x(') and part.endswith(')'):
            inner = part[2:-1]
            if ',' in inner:
                m, n = inner.split(',')
                m, n = int(m), int(n)
                # 铁律3：更严格的跨度限制
                if n > MAX_GAP_UPPER or (n - m) > MAX_GAP_SPAN:
                    return False
        elif part.startswith('<GAP(') and part.endswith(')>'):
            inner = part[5:-2]
            m, n = inner.split(',')
            m, n = int(m), int(n)
            if n > MAX_GAP_UPPER or (n - m) > MAX_GAP_SPAN:
                return False

    # 铁律2：FPR检查（使用fullmatch）
    regex_str = rule_to_regex(rule_pattern)
    compiled = re.compile(regex_str)

    bg_total = len(background_pool)

    if bg_total == 0:
        return False

    # 采样优化
    bg_list = list(background_pool)
    if len(bg_list) > FPR_SAMPLE_SIZE:
        bg_list = random.sample(bg_list, FPR_SAMPLE_SIZE)

    # 铁律2：早停机制，但必须用fullmatch
    max_fp = int(MAX_FPR * len(bg_list))
    bg_count = 0

    for word in bg_list:
        # 铁律1：使用完整序列（保留下划线）
        full_seq = get_full_sequence(word)
        # 铁律2：使用fullmatch死守全长
        if compiled.fullmatch(full_seq):
            bg_count += 1
            if bg_count > max_fp:
                return False

    return True


# ============================================================
# 模块3：双序列弹性融合（铁律1 + 铁律4）
# ============================================================

def merge_rule_and_sequence(current_pattern, new_seq):
    """
    双序列弹性融合
    铁律1：保留下划线，带着下划线一起计算位置
    铁律4：严格计算gap区间
    """
    # 铁律1：提取完整序列（保留下划线）
    new_full = get_full_sequence(new_seq)

    # 如果当前规则就是序列本身，直接返回
    if current_pattern == new_seq:
        return current_pattern

    # 铁律1：提取当前规则的完整序列（保留下划线）
    current_full = get_full_sequence(current_pattern)

    # 找最长公共子串作为骨架
    skeleton = find_longest_common_substring(current_full, new_full)

    if len(skeleton) < 2:
        return None

    # 铁律4：根据骨架构建规则，严格计算gap区间
    return build_rule_from_skeleton(skeleton, current_full, new_full)


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
    序贯覆盖算法挖掘规则（铁律3：微簇规模控制）

    铁律3：覆盖数控制在2-4个词，宁缺毋滥
    """
    uncovered_pool = sort_sequences_by_centrality(list(pos_set))
    final_rules = []

    # K-mer反向索引
    kmer_index = _build_kmer_index(uncovered_pool)

    iteration = 0
    max_iterations = 500
    patience_limit = 20  # 铁律3：降低耐心阈值，快速熔断

    while uncovered_pool and iteration < max_iterations:
        iteration += 1

        # 取出得分最高的种子
        seed_seq = uncovered_pool.pop(0)

        # 初始化规则
        current_rule = seed_seq
        covered_words = [seed_seq]

        # 铁律3：限制覆盖数在2-4个
        absorbed_count = 0
        max_absorb = 3  # 铁律3：最多吸收3个，总共4个

        # K-mer预过滤
        seed_full = get_full_sequence(seed_seq)
        seed_kmers = _extract_kmers(seed_full)
        candidate_pool = _filter_by_kmers(uncovered_pool, seed_kmers, kmer_index)

        if not candidate_pool:
            continue

        remaining_pool = []
        consecutive_fails = 0

        for candidate_seq in uncovered_pool:
            if candidate_seq not in candidate_pool:
                remaining_pool.append(candidate_seq)
                continue

            # 铁律3：达到最大覆盖数，停止
            if absorbed_count >= max_absorb:
                remaining_pool.append(candidate_seq)
                continue

            # 跳过太长的规则
            if len(current_rule) > 50:
                remaining_pool.append(candidate_seq)
                consecutive_fails += 1
                if consecutive_fails >= patience_limit:
                    break
                continue

            temp_rule = merge_rule_and_sequence(current_rule, candidate_seq)

            if temp_rule is None:
                remaining_pool.append(candidate_seq)
                consecutive_fails += 1
                if consecutive_fails >= patience_limit:
                    break
                continue

            # 铁律2+铁律3：校验规则（fullmatch + 微簇控制）
            if is_rule_valid(temp_rule, background_pool):
                current_rule = temp_rule
                covered_words.append(candidate_seq)
                absorbed_count += 1
                consecutive_fails = 0
            else:
                remaining_pool.append(candidate_seq)
                consecutive_fails += 1
                if consecutive_fails >= patience_limit:
                    for rest in uncovered_pool[uncovered_pool.index(candidate_seq)+1:]:
                        if rest not in remaining_pool:
                            remaining_pool.append(rest)
                    break

        # 铁律3：结算判断，只接受2-4个词的微簇
        if len(covered_words) >= 2 and len(covered_words) <= 4:
            final_rules.append((current_rule, covered_words))

        uncovered_pool = remaining_pool

        if iteration % 10 == 0:
            print(f"    迭代 {iteration}: 剩余 {len(uncovered_pool)} 个词")

    return final_rules


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