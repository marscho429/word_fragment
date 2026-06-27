"""
蛋白词预处理模块
处理下划线、标准化氨基酸序列等
"""
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import re

# 导入ALL_AMINO_ACIDS常量
ALL_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')


@dataclass
class ProcessedWord:
    """处理后的蛋白词"""
    original: str              # 原始蛋白词
    amino_sequence: List[str]  # 氨基酸序列（不含下划线）
    has_underscore: bool       # 是否包含下划线
    underscore_positions: List[Tuple[int, int]]  # 下划线在氨基酸序列中的位置
    length: int                # 氨基酸序列长度
    
    def __hash__(self):
        return hash(self.original)
    
    def __eq__(self, other):
        return self.original == other.original


def preprocess_word(word: str) -> ProcessedWord:
    """
    预处理单个蛋白词
    处理下划线，提取氨基酸序列
    """
    # 分割蛋白词，识别下划线
    parts = word.split('_')
    
    amino_sequence = []
    underscore_positions = []
    has_underscore = len(parts) > 1
    
    # 解析氨基酸和下划线位置
    current_pos = 0
    last_was_underscore = False
    
    for i, part in enumerate(parts):
        if part == '':
            # 这是下划线（空字符串部分）
            if current_pos > 0 and i < len(parts) - 1:
                # 下划线在中间或末尾
                if not underscore_positions or underscore_positions[-1][1] != current_pos:
                    underscore_positions.append((current_pos, current_pos))
                last_was_underscore = True
        else:
            # 这是氨基酸部分
            # 验证氨基酸
            valid_amino_acids = [aa for aa in part if aa in ALL_AMINO_ACIDS]
            amino_sequence.extend(valid_amino_acids)
            
            # 如果前面有下划线，更新下划线位置
            if last_was_underscore and underscore_positions:
                underscore_positions[-1] = (underscore_positions[-1][0], current_pos)
            
            current_pos += len(valid_amino_acids)
            last_was_underscore = False
    
    return ProcessedWord(
        original=word,
        amino_sequence=amino_sequence,
        has_underscore=has_underscore,
        underscore_positions=underscore_positions,
        length=len(amino_sequence)
    )


def preprocess_words(words: Set[str]) -> List[ProcessedWord]:
    """
    预处理一组蛋白词
    """
    return [preprocess_word(word) for word in words]


def group_words_by_length(words: List[ProcessedWord]) -> Dict[int, List[ProcessedWord]]:
    """
    按长度分组蛋白词
    """
    groups = {}
    for word in words:
        length = word.length
        if length not in groups:
            groups[length] = []
        groups[length].append(word)
    return groups


def find_similar_words(word: ProcessedWord, 
                       other_words: List[ProcessedWord],
                       min_similarity: float = 0.7) -> List[ProcessedWord]:
    """
    找到相似度高的蛋白词
    使用简单的重叠度计算
    """
    similar = []
    
    for other in other_words:
        if other.original == word.original:
            continue
        
        # 计算氨基酸重叠度
        overlap = calculate_amino_overlap(word.amino_sequence, other.amino_sequence)
        
        if overlap >= min_similarity:
            similar.append(other)
    
    return similar


def calculate_amino_overlap(seq1: List[str], seq2: List[str]) -> float:
    """
    计算两个氨基酸序列的重叠度
    使用最长公共子序列(LCS)方法
    """
    if not seq1 or not seq2:
        return 0.0
    
    # 计算LCS长度
    lcs_length = longest_common_subsequence_length(seq1, seq2)
    
    # 重叠度 = LCS长度 / max(len1, len2)
    max_length = max(len(seq1), len(seq2))
    return lcs_length / max_length if max_length > 0 else 0.0


def longest_common_subsequence_length(seq1: List[str], seq2: List[str]) -> int:
    """
    计算最长公共子序列长度
    使用动态规划
    """
    m, n = len(seq1), len(seq2)
    
    # 创建DP表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def extract_common_pattern(words: List[ProcessedWord]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    从一组蛋白词中提取公共模式
    返回：(公共氨基酸序列, 差异位置列表)
    """
    if not words:
        return [], []
    
    if len(words) == 1:
        return words[0].amino_sequence, []
    
    # 使用第一个词作为基准
    base_seq = words[0].amino_sequence
    common_positions = []
    differences = []
    
    # 对每个位置检查是否所有词都相同
    for pos in range(len(base_seq)):
        amino_acids_at_pos = set()
        
        for word in words:
            if pos < word.length:
                amino_acids_at_pos.add(word.amino_sequence[pos])
        
        if len(amino_acids_at_pos) == 1:
            # 所有词在这个位置都相同
            common_positions.append(pos)
        else:
            # 存在差异
            differences.append((pos, amino_acids_at_pos))
    
    # 构建公共序列
    common_sequence = [base_seq[pos] for pos in common_positions]
    
    return common_sequence, differences


def cluster_words_by_pattern(words: List[ProcessedWord]) -> Dict[str, List[ProcessedWord]]:
    """
    根据模式聚类蛋白词
    简单版本：按氨基酸组成聚类
    """
    clusters = {}
    
    for word in words:
        # 创建简单的模式签名：氨基酸组成
        amino_set = tuple(sorted(set(word.amino_sequence)))
        pattern_key = f"{len(word.amino_sequence)}_{amino_set}"
        
        if pattern_key not in clusters:
            clusters[pattern_key] = []
        clusters[pattern_key].append(word)
    
    return clusters


def find_word_pairs_with_high_overlap(words: List[ProcessedWord],
                                      min_overlap: float = 0.8) -> List[Tuple[ProcessedWord, ProcessedWord, float]]:
    """
    找到高重叠度的蛋白词对
    返回：(word1, word2, overlap_score)列表
    """
    pairs = []
    
    # 按长度分组，减少计算量
    length_groups = group_words_by_length(words)
    
    # 只比较相近长度的词（长度差不超过3）
    for length, group in length_groups.items():
        # 检查相近长度组
        for other_length in range(length - 3, length + 4):
            if other_length in length_groups:
                other_group = length_groups[other_length]
                
                # 组内和组间比较
                for word1 in group:
                    for word2 in other_group:
                        if word1.original < word2.original:  # 避免重复
                            overlap = calculate_amino_overlap(
                                word1.amino_sequence,
                                word2.amino_sequence
                            )
                            if overlap >= min_overlap:
                                pairs.append((word1, word2, overlap))
    
    return pairs


def get_negative_sample_pool(all_words: Set[str], 
                             fragment_words: Set[str],
                             sample_size: int = 10000) -> List[str]:
    """
    生成负样本池
    从所有蛋白词中选择不在当前fragment中的词
    """
    negative_words = [word for word in all_words if word not in fragment_words]
    
    # 如果负样本数量少于要求，返回所有负样本
    if len(negative_words) <= sample_size:
        return negative_words
    
    # 随机采样
    import random
    return random.sample(negative_words, sample_size)