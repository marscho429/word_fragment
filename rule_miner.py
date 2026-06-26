#!/usr/bin/env python3
"""
蛋白词-分子片段规则挖掘算法
基于"代表性种子+序贯覆盖"的基序挖掘 (v10.0)

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
MAX_GAP_SPAN = 4      # 最大Gap跨度
MAX_GAP_UPPER = 5     # 单个Gap最大上限
MAX_WILDCARD_RATIO = 0.5  # 最大通配比例

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

def extract_solid_sequence(word):
    """提取实心氨基酸序列（保留下划线用于后续处理）"""
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


def rule_to_regex(rule_pattern):
    """将规则模式转换为正则表达式，支持 GAP(m,n) 格式"""
    parts = rule_pattern.split('-')
    regex_parts = []
    for part in parts:
        if part.startswith('x(') and part.endswith(')'):
            inner = part[2:-1]
            if ',' in inner:
                m, n = inner.split(',')
            else:
                m = n = inner
            regex_parts.append(f'[A-Z_]{{{m},{n}}}')
        elif part.startswith('<GAP(') and part.endswith(')>'):
            inner = part[5:-2]
            m, n = inner.split(',')
            regex_parts.append(f'[A-Z_]{{{m},{n}}}')
        elif part == 'x':
            regex_parts.append('[A-Z_]')
        elif part == '_':
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
    return block


# ============================================================
# 模块1: 种子优先级排序 (Representative Seeding)
# ============================================================

def sort_sequences_by_centrality(seq_list):
    """
    按中心度得分降序排列序列

    1. 统计所有3-mer的全局频次
    2. 为每条序列计算中心度得分（包含的3-mer频次之和）
    3. 返回降序排列
    """
    # 提取所有实心序列
    solid_seqs = [''.join(c for c in seq if c in AA_LETTERS) for seq in seq_list]

    # 统计所有3-mer频次
    kmer_counts = defaultdict(int)
    for solid in solid_seqs:
        for i in range(len(solid) - 2):
            kmer = solid[i:i+3]
            kmer_counts[kmer] += 1

    # 计算每条序列的中心度得分
    seq_scores = []
    for seq, solid in zip(seq_list, solid_seqs):
        score = 0
        for i in range(len(solid) - 2):
            kmer = solid[i:i+3]
            score += kmer_counts[kmer]
        seq_scores.append((seq, score))

    # 降序排列
    seq_scores.sort(key=lambda x: x[1], reverse=True)
    return [seq for seq, _ in seq_scores]


# ============================================================
# 模块2: 严苛的规则校验器 (The Iron Gates)
# ============================================================

def is_rule_valid(rule_pattern, background_pool):
    """
    严苛的规则校验，必须同时满足3个条件

    1. 宏观结构通配比例 <= 50%
    2. 微观弹簧跨度限制 (Gap span <= 4, Gap upper <= 5)
    3. 极低假阳性 (FPR <= 0.005)
    """
    parts = rule_pattern.split('-')

    # 条件1: 通配比例检查
    solid_blocks = 0
    wildcard_blocks = 0

    for part in parts:
        if part.startswith('x(') or part.startswith('<GAP(') or part == 'x' or part == '_':
            wildcard_blocks += 1
        else:
            solid_blocks += 1

    total_blocks = solid_blocks + wildcard_blocks
    if total_blocks == 0:
        return False

    wildcard_ratio = wildcard_blocks / total_blocks
    if wildcard_ratio > MAX_WILDCARD_RATIO:
        return False

    # 条件2: Gap跨度限制
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

    # 条件3: FPR检查
    regex_str = rule_to_regex(rule_pattern)
    compiled = re.compile(regex_str)

    bg_count = 0
    bg_total = len(background_pool)

    if bg_total == 0:
        return False

    for word in background_pool:
        solid_seq = extract_solid_sequence(word)
        if compiled.fullmatch(solid_seq):
            bg_count += 1

    fpr = bg_count / bg_total
    return fpr <= MAX_FPR


# ============================================================
# 模块3: 双序列弹性融合 (Elastic Merge Engine)
# ============================================================

def merge_rule_and_sequence(current_pattern, new_seq):
    """
    使用DP比对将新序列融合到当前规则中

    1. 将规则和序列转换为对齐矩阵
    2. 下划线允许0惩罚吞并错配残基
    3. 垂直提取共识
    4. 强制头尾闭环
    """
    # 解析当前规则为token列表
    rule_tokens = parse_rule_tokens(current_pattern)

    # 提取新序列的实心氨基酸
    new_solid = extract_solid_sequence(new_seq)

    # DP比对
    alignment = dp_align_rule_and_sequence(rule_tokens, new_solid)

    # 垂直提取共识
    new_tokens = extract_consensus_from_alignment(alignment)

    # 计算头尾闭环
    left_gap, right_gap = calculate_terminal_gaps(alignment)

    # 构建新规则
    new_parts = []
    if left_gap[1] > 0:
        new_parts.append(f'x({left_gap[0]},{left_gap[1]})')
    new_parts.extend(new_tokens)
    if right_gap[1] > 0:
        new_parts.append(f'x({right_gap[0]},{right_gap[1]})')

    return '-'.join(new_parts)


def parse_rule_tokens(rule_pattern):
    """将规则模式解析为token列表"""
    parts = rule_pattern.split('-')
    tokens = []

    for part in parts:
        if part.startswith('x(') and part.endswith(')'):
            tokens.append(('gap', part))
        elif part.startswith('<GAP(') and part.endswith(')>'):
            tokens.append(('gap', part))
        elif part == 'x' or part == '_':
            tokens.append(('gap', part))
        else:
            tokens.append(('solid', part))

    return tokens


def dp_align_rule_and_sequence(rule_tokens, sequence):
    """
    使用DP比对规则token和序列

    下划线和gap token可以以0惩罚吞并任意字符
    """
    m = len(rule_tokens)
    n = len(sequence)

    # DP矩阵: dp[i][j] = (score, parent_i, parent_j)
    dp = [[(-float('inf'), -1, -1) for _ in range(n + 1)] for _ in range(m + 1)]
    dp[0][0] = (0, -1, -1)

    # 初始化第一行和第一列
    for i in range(1, m + 1):
        token_type, token_val = rule_tokens[i-1]
        if token_type == 'gap':
            dp[i][0] = (0, i-1, 0)  # gap可以匹配空
        else:
            break

    for j in range(1, n + 1):
        dp[0][j] = (0, 0, j-1)  # 序列可以不匹配任何token

    # 填充DP矩阵
    for i in range(1, m + 1):
        token_type, token_val = rule_tokens[i-1]

        for j in range(1, n + 1):
            char = sequence[j-1]

            # 匹配分数
            if token_type == 'gap':
                match_score = 0
            else:
                match_score = check_token_match(token_val, char)

            # 三个方向
            match = dp[i-1][j-1][0] + match_score
            delete = dp[i-1][j][0] - 1
            insert = dp[i][j-1][0] - 1

            if match >= delete and match >= insert:
                dp[i][j] = (match, i-1, j-1)
            elif delete >= insert:
                dp[i][j] = (delete, i-1, j)
            else:
                dp[i][j] = (insert, i, j-1)

    # 回溯路径
    i, j = m, n
    path = []

    while i > 0 or j > 0:
        token = rule_tokens[i-1] if i > 0 else None
        char = sequence[j-1] if j > 0 else None

        if i > 0 and j > 0 and dp[i][j][1] == i-1 and dp[i][j][2] == j-1:
            path.append(('match', token, char))
            i, j = i-1, j-1
        elif i > 0 and dp[i][j][1] == i-1:
            path.append(('delete', token, None))
            i -= 1
        else:
            path.append(('insert', None, char))
            j -= 1

    path.reverse()
    return path


def check_token_match(token_val, char):
    """检查token是否匹配字符"""
    if len(token_val) == 1 and token_val == char:
        return 2  # 完全匹配
    elif len(token_val) == 1:
        g1 = get_aa_group(token_val)
        g2 = get_aa_group(char)
        if g1 and g2 and token_val in g2 and char in g1:
            return 1  # 同族匹配
    elif token_val.startswith('[') and token_val.endswith(']'):
        inner = token_val[1:-1]
        if inner in SEMANTIC_TO_REGEX:
            regex = SEMANTIC_TO_REGEX[inner]
            if re.match(regex, char):
                return 1
        else:
            if char in inner:
                return 1
    return -1  # 不匹配


def extract_consensus_from_alignment(alignment):
    """
    从对齐中提取共识token（简化版）

    相同字母保留字母，同族替换为[族]，冲突替换为x
    """
    consensus_tokens = []

    # 简化处理：直接按操作类型处理
    for op, token, char in alignment:
        if op == 'match' and token is not None:
            # 提取token中的氨基酸
            if isinstance(token, tuple):
                token_type, token_val = token
                if token_type == 'solid':
                    consensus_tokens.append(token_val)
                else:
                    consensus_tokens.append('x')
            elif len(token) == 1 and token in AA_LETTERS:
                # 合并char和token
                if token == char:
                    consensus_tokens.append(token)
                else:
                    # 检查是否同族
                    g1 = get_aa_group(token)
                    g2 = get_aa_group(char)
                    if g1 and g2 and token in g2 and char in g1:
                        consensus_tokens.append(_aa_set_to_block({token, char}))
                    else:
                        consensus_tokens.append('x')
            else:
                consensus_tokens.append(token)
        elif op == 'insert' and char is not None:
            consensus_tokens.append(char)
        elif op == 'delete' and token is not None:
            if isinstance(token, tuple):
                token_type, token_val = token
                if token_type == 'solid':
                    consensus_tokens.append(token_val)
                else:
                    consensus_tokens.append('x')
            else:
                consensus_tokens.append(token)

    # 合并相邻的单字符
    if not consensus_tokens:
        return ['x']

    fused_tokens = []
    current_chars = set()

    for token in consensus_tokens:
        if len(token) == 1 and token in AA_LETTERS:
            current_chars.add(token)
        else:
            if current_chars:
                fused_tokens.append(_aa_set_to_block(current_chars))
                current_chars = set()
            fused_tokens.append(token)

    if current_chars:
        fused_tokens.append(_aa_set_to_block(current_chars))

    return fused_tokens


def calculate_terminal_gaps(alignment):
    """
    计算头尾闭环的GAP范围

    返回: ((min_L, max_L), (min_R, max_R))
    """
    # 找出第一个匹配的位置
    first_match_idx = 0
    for i, (op, token, char) in enumerate(alignment):
        if op == 'match':
            first_match_idx = i
            break

    # 找出最后一个匹配的位置
    last_match_idx = len(alignment) - 1
    for i in range(len(alignment) - 1, -1, -1):
        op, token, char = alignment[i]
        if op == 'match':
            last_match_idx = i
            break

    # 计算左侧悬垂
    left_gap_chars = []
    for i in range(first_match_idx):
        op, token, char = alignment[i]
        if char is not None:
            left_gap_chars.append(char)

    # 计算右侧悬垂
    right_gap_chars = []
    for i in range(last_match_idx + 1, len(alignment)):
        op, token, char = alignment[i]
        if char is not None:
            right_gap_chars.append(char)

    # 简化计算：使用字符数量
    min_L = len(left_gap_chars)
    max_L = len(left_gap_chars)
    min_R = len(right_gap_chars)
    max_R = len(right_gap_chars)

    return ((min_L, max_L), (min_R, max_R))


# ============================================================
# 模块4: 主循环 (Sequential Covering Pipeline)
# ============================================================

def mine_rules(pos_set, background_pool):
    """
    序贯覆盖算法挖掘规则（极速优化版）

    优化策略：
    1. K-mer预过滤：只尝试与种子共享至少一个3-mer的候选
    2. 短路校验：先算通配比例，再算FPR
    3. 耐心阈值：连续失败200次后放弃种子

    1. 按中心度排序正样本
    2. 依次取出种子，尝试吸收更多样本
    3. 生成的规则必须通过严苛校验
    """
    uncovered_pool = sort_sequences_by_centrality(list(pos_set))
    final_rules = []

    # 优化1: 建立K-mer反向索引
    kmer_index = _build_kmer_index(uncovered_pool)

    iteration = 0
    max_iterations = 1000  # 防止无限循环
    patience_limit = 200   # 优化3: 连续失败阈值

    while uncovered_pool and iteration < max_iterations:
        iteration += 1

        # 取出得分最高的种子
        seed_seq = uncovered_pool.pop(0)

        # 初始化规则
        current_rule = seed_seq
        covered_words = [seed_seq]

        # 尝试吸收更多序列
        absorbed_count = 0
        max_absorb = 100  # 限制单次最大吸收数

        # 优化1: 通过K-mer预过滤候选序列
        seed_kmers = _extract_kmers(extract_solid_sequence(seed_seq))
        candidate_pool = _filter_by_kmers(uncovered_pool, seed_kmers, kmer_index)

        # 如果没有候选，直接跳过
        if not candidate_pool:
            continue

        remaining_pool = []
        consecutive_fails = 0  # 优化3: 连续失败计数器

        for candidate_seq in uncovered_pool:
            # 优化1: 跳过不共享任何3-mer的候选
            if candidate_seq not in candidate_pool:
                remaining_pool.append(candidate_seq)
                continue

            if absorbed_count >= max_absorb:
                remaining_pool.append(candidate_seq)
                continue

            temp_rule = merge_rule_and_sequence(current_rule, candidate_seq)

            if is_rule_valid(temp_rule, background_pool):
                # 吸收成功
                current_rule = temp_rule
                covered_words.append(candidate_seq)
                absorbed_count += 1
                consecutive_fails = 0  # 重置计数器
            else:
                remaining_pool.append(candidate_seq)
                consecutive_fails += 1

                # 优化3: 连续失败超过阈值，放弃该种子
                if consecutive_fails >= patience_limit:
                    # 将剩余候选全部加入remaining_pool
                    for rest in uncovered_pool[uncovered_pool.index(candidate_seq)+1:]:
                        if rest not in remaining_pool:
                            remaining_pool.append(rest)
                    break

        # 结算判断
        if len(covered_words) >= 2:
            # 合格的泛化规则
            final_rules.append((current_rule, covered_words))
        # 否则丢弃孤儿词

        # 更新未覆盖池和反向索引
        uncovered_pool = remaining_pool
        kmer_index = _build_kmer_index(uncovered_pool)

        if iteration % 10 == 0:
            print(f"    迭代 {iteration}: 剩余 {len(uncovered_pool)} 个词")

    return final_rules


def _build_kmer_index(seq_list, k=3):
    """
    构建K-mer反向索引

    返回: dict(kmer -> set(sequences))
    """
    index = defaultdict(set)
    for seq in seq_list:
        solid = extract_solid_sequence(seq)
        for i in range(len(solid) - k + 1):
            kmer = solid[i:i+k]
            index[kmer].add(seq)
    return index


def _extract_kmers(seq, k=3):
    """提取序列中的所有K-mer"""
    kmers = set()
    for i in range(len(seq) - k + 1):
        kmers.add(seq[i:i+k])
    return kmers


def _filter_by_kmers(pool, seed_kmers, kmer_index):
    """
    根据K-mer预过滤候选序列

    返回: 与seed共享至少一个3-mer的候选集合
    """
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

        # 计算FPR
        regex_str = rule_to_regex(rule_pattern)
        compiled = re.compile(regex_str)

        bg_count = 0
        for word in B:
            solid_seq = extract_solid_sequence(word)
            if compiled.fullmatch(solid_seq):
                bg_count += 1

        fpr = bg_count / len(B) if B else 0.0

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
    parser = argparse.ArgumentParser(description='蛋白词规则挖掘 (v10.0)')
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