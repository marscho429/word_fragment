"""
规则验证模块
验证规则的有效性，计算FPR等
"""
from typing import List, Set, Dict, Tuple
import re

from core import GeneralizationRule, RuleElement, RuleElementType, TIERED_SEMANTIC_GROUPS
from preprocess import preprocess_word, ProcessedWord


def compile_rule_to_regex(rule: GeneralizationRule) -> re.Pattern:
    """
    将泛化规则编译为正则表达式
    """
    pattern_str = rule.to_regex_pattern()
    try:
        return re.compile(f"^{pattern_str}$")
    except re.error:
        # 如果编译失败，返回一个空匹配的模式
        return re.compile("^$")


def match_word_with_rule(word: str, rule: GeneralizationRule) -> bool:
    """
    检查规则是否能匹配蛋白词
    """
    # 预处理蛋白词
    processed = preprocess_word(word)
    
    # 构建匹配序列（将下划线转换为任意氨基酸）
    match_sequence = build_match_sequence(processed)
    
    # 编译规则
    regex_pattern = compile_rule_to_regex(rule)
    
    # 尝试匹配
    return bool(regex_pattern.match(match_sequence))


def build_match_sequence(processed: ProcessedWord) -> str:
    """
    构建用于匹配的序列字符串
    处理下划线：转换为任意氨基酸序列的正则表达式
    """
    parts = processed.original.split('_')
    result_parts = []
    
    for i, part in enumerate(parts):
        if part == '':
            if i > 0 and i < len(parts) - 1:
                # 中间的下划线：一个或多个任意氨基酸
                result_parts.append('[ACDEFGHIKLMNPQRSTVWY]+')
            elif i == 0 and len(parts) > 1:
                # 开头的下划线
                result_parts.append('[ACDEFGHIKLMNPQRSTVWY]+')
            elif i == len(parts) - 1 and len(parts) > 1:
                # 结尾的下划线
                result_parts.append('[ACDEFGHIKLMNPQRSTVWY]+')
        else:
            result_parts.append(part)
    
    return ''.join(result_parts)


def validate_rule_against_original_words(rule: GeneralizationRule,
                                         original_words: Set[str]) -> bool:
    """
    验证规则是否能匹配所有原始蛋白词
    """
    for word in original_words:
        if not match_word_with_rule(word, rule):
            return False
    return True


def calculate_fpr_for_rule(rule: GeneralizationRule,
                          negative_samples: List[str]) -> float:
    """
    计算单条规则的假阳性率
    """
    if not negative_samples:
        return 0.0
    
    false_positives = 0
    
    for word in negative_samples:
        if match_word_with_rule(word, rule):
            false_positives += 1
    
    return false_positives / len(negative_samples)


def calculate_fpr_for_rule_set(rules: List[GeneralizationRule],
                               negative_samples: List[str]) -> float:
    """
    计算规则集合的整体假阳性率
    """
    if not negative_samples:
        return 0.0
    
    # 统计有多少负样本至少匹配了一条规则
    false_positives = 0
    
    for word in negative_samples:
        matched = False
        for rule in rules:
            if match_word_with_rule(word, rule):
                matched = True
                break
        
        if matched:
            false_positives += 1
    
    return false_positives / len(negative_samples)


def evaluate_rule_quality(rule: GeneralizationRule,
                         negative_samples: List[str]) -> Dict[str, float]:
    """
    评估规则质量
    返回多个质量指标
    """
    # 计算各项指标
    fpr = calculate_fpr_for_rule(rule, negative_samples)
    wildcard_ratio = rule.get_wildcard_ratio()
    priority_score = rule.get_priority_score()
    coverage = rule.covered_count
    
    # 综合质量分数
    quality_score = (
        priority_score * 0.4 +
        coverage * 0.3 +
        (1 - fpr) * 100 * 0.2 +
        (1 - wildcard_ratio) * 10 * 0.1
    )
    
    return {
        'fpr': fpr,
        'wildcard_ratio': wildcard_ratio,
        'priority_score': priority_score,
        'coverage': coverage,
        'quality_score': quality_score
    }


def validate_rule_set(rules: List[GeneralizationRule],
                     target_words: Set[str],
                     negative_samples: List[str],
                     max_fpr: float = 0.1,
                     min_coverage: float = 0.85) -> Tuple[bool, Dict[str, float]]:
    """
    验证整个规则集合
    """
    # 检查覆盖率
    covered_words = set()
    for rule in rules:
        covered_words.update(rule.covered_words)
    
    coverage = len(covered_words) / len(target_words) if target_words else 0
    
    # 检查FPR
    fpr = calculate_fpr_for_rule_set(rules, negative_samples)
    
    # 检查规则是否能匹配原始词
    all_match = True
    for rule in rules:
        if not validate_rule_against_original_words(rule, rule.covered_words):
            all_match = False
            break
    
    # 检查wildcard比例
    avg_wildcard_ratio = sum(r.get_wildcard_ratio() for r in rules) / len(rules) if rules else 0
    
    # 判断是否通过验证
    passed = (
        coverage >= min_coverage and
        fpr <= max_fpr and
        all_match and
        avg_wildcard_ratio <= 0.5
    )
    
    metrics = {
        'coverage': coverage,
        'fpr': fpr,
        'all_match': all_match,
        'avg_wildcard_ratio': avg_wildcard_ratio,
        'rule_count': len(rules)
    }
    
    return passed, metrics


def filter_rules_by_quality(rules: List[GeneralizationRule],
                           negative_samples: List[str],
                           max_fpr: float = 0.05,
                           min_coverage: int = 2) -> List[GeneralizationRule]:
    """
    根据质量指标过滤规则
    """
    filtered = []
    
    for rule in rules:
        # 基本过滤条件
        if rule.covered_count < min_coverage:
            continue
        
        if rule.get_wildcard_ratio() > 0.5:
            continue
        
        # 计算FPR
        fpr = calculate_fpr_for_rule(rule, negative_samples)
        if fpr > max_fpr:
            continue
        
        # 添加到过滤后的列表
        rule_metrics = evaluate_rule_quality(rule, negative_samples)
        rule.fpr = fpr  # 添加FPR属性（需要扩展dataclass）
        filtered.append(rule)
    
    # 按质量分数排序
    filtered.sort(key=lambda r: r.get_priority_score(), reverse=True)
    
    return filtered


def compute_coverage_metrics(rules: List[GeneralizationRule],
                            target_words: Set[str]) -> Dict[str, float]:
    """
    计算覆盖率相关指标
    """
    covered = set()
    word_to_rule_count = {}
    
    for rule in rules:
        covered.update(rule.covered_words)
        for word in rule.covered_words:
            word_to_rule_count[word] = word_to_rule_count.get(word, 0) + 1
    
    coverage_ratio = len(covered) / len(target_words) if target_words else 0
    
    # 计算平均每条规则覆盖的词数
    avg_coverage_per_rule = sum(r.covered_count for r in rules) / len(rules) if rules else 0
    
    # 计算平均每个词被多少条规则覆盖
    avg_rules_per_word = sum(word_to_rule_count.values()) / len(word_to_rule_count) if word_to_rule_count else 0
    
    return {
        'coverage_ratio': coverage_ratio,
        'total_covered': len(covered),
        'total_target': len(target_words),
        'avg_coverage_per_rule': avg_coverage_per_rule,
        'avg_rules_per_word': avg_rules_per_word
    }


class RuleValidator:
    """规则验证器"""
    
    def __init__(self, negative_pool: List[str]):
        self.negative_pool = negative_pool
    
    def validate_single_rule(self, rule: GeneralizationRule) -> Dict[str, float]:
        """验证单条规则"""
        return evaluate_rule_quality(rule, self.negative_pool)
    
    def validate_rule_set(self, 
                         rules: List[GeneralizationRule],
                         target_words: Set[str]) -> Tuple[bool, Dict]:
        """验证规则集合"""
        return validate_rule_set(rules, target_words, self.negative_pool)
    
    def filter_rules(self, rules: List[GeneralizationRule]) -> List[GeneralizationRule]:
        """过滤规则"""
        return filter_rules_by_quality(rules, self.negative_pool)