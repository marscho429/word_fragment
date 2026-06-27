"""
泛化规则生成模块
核心算法：聚类 + MSA + 属性类泛化
"""
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import difflib

from core import (
    RuleElement, GeneralizationRule, RuleElementType, TIERED_SEMANTIC_GROUPS,
    ALL_AMINO_ACIDS, get_best_property_for_amino_acids,
    create_exact_element, create_property_element, create_optional_element,
    create_wildcard_fixed_element, create_wildcard_range_element,
    create_wildcard_any_element, create_choice_element
)
from preprocess import ProcessedWord, preprocess_word


def group_words_by_similarity(words: List[ProcessedWord],
                              min_group_size: int = 2,
                              max_group_size: int = 3,
                              similarity_threshold: float = 0.8) -> List[List[ProcessedWord]]:
    """
    按相似度将词分组
    使用前缀、后缀和编辑距离综合判断
    """
    groups = []
    used = set()
    
    sorted_words = sorted(words, key=lambda w: len(w.amino_sequence))
    
    for i, word1 in enumerate(sorted_words):
        if word1.original in used:
            continue
        
        group = [word1]
        used.add(word1.original)
        
        seq1 = ''.join(word1.amino_sequence)
        
        for j in range(i + 1, len(sorted_words)):
            word2 = sorted_words[j]
            if word2.original in used:
                continue
            if len(group) >= max_group_size:
                break
            
            seq2 = ''.join(word2.amino_sequence)
            
            # 计算相似度
            similarity = calculate_similarity(seq1, seq2)
            
            if similarity >= similarity_threshold:
                group.append(word2)
                used.add(word2.original)
                seq1 = seq2
        
        if len(group) >= min_group_size:
            groups.append(group)
    
    return groups


def calculate_similarity(seq1: str, seq2: str) -> float:
    """
    计算两个序列的相似度
    综合前缀、后缀和编辑距离
    """
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    
    if max_len == 0:
        return 0.0
    
    # 前缀相似度
    prefix_len = 0
    for a, b in zip(seq1, seq2):
        if a == b:
            prefix_len += 1
        else:
            break
    
    # 后缀相似度
    suffix_len = 0
    for a, b in zip(reversed(seq1), reversed(seq2)):
        if a == b:
            suffix_len += 1
        else:
            break
    
    # 编辑距离相似度
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    edit_sim = matcher.ratio()
    
    # 综合相似度
    prefix_score = prefix_len / max_len * 0.3
    suffix_score = suffix_len / max_len * 0.3
    edit_score = edit_sim * 0.4
    
    return prefix_score + suffix_score + edit_score


def align_word_group(words: List[ProcessedWord]) -> List[List[str]]:
    """
    对一组词做多序列对齐
    使用居中对齐策略，短序列两边填充gap
    """
    sequences = [list(''.join(w.amino_sequence)) for w in words]
    max_len = max(len(s) for s in sequences)
    
    aligned = []
    for seq in sequences:
        padding = max_len - len(seq)
        left_pad = padding // 2
        right_pad = padding - left_pad
        
        aligned_seq = ['-'] * left_pad + seq + ['-'] * right_pad
        aligned.append(aligned_seq)
    
    return aligned


def build_rule_from_alignment(aligned_sequences: List[List[str]],
                              word_originals: Set[str]) -> GeneralizationRule:
    """
    从对齐结果生成泛化规则
    策略：
    1. 每个位置尝试匹配性质类（优先Tier1）
    2. 确保规则中至少包含一定比例的性质类
    3. 对于无法匹配性质类的位置，使用更精细的规则
    4. gap位置用可选标记
    """
    if not aligned_sequences:
        return GeneralizationRule(elements=[], covered_words=word_originals, covered_count=len(word_originals))
    
    max_pos = len(aligned_sequences[0])
    elements = []
    total_words = len(aligned_sequences)
    
    strict_mode = total_words > 15
    
    for pos in range(max_pos):
        amino_acids = set()
        amino_counts = defaultdict(int)
        gap_count = 0
        
        for seq in aligned_sequences:
            if pos < len(seq) and seq[pos] != '-':
                aa = seq[pos]
                amino_acids.add(aa)
                amino_counts[aa] += 1
            else:
                gap_count += 1
        
        if not amino_acids:
            continue
        
        is_optional = gap_count > 0 and gap_count < total_words * 0.5
        
        if is_optional:
            if amino_counts:
                most_common = max(amino_counts, key=amino_counts.get)
                count = amino_counts[most_common]
                
                if count >= total_words * 0.5:
                    prop = find_property_for_single_amino(most_common)
                    if prop and not strict_mode:
                        elements.append(RuleElement(type=RuleElementType.OPTIONAL, value=prop))
                    else:
                        elements.append(RuleElement(type=RuleElementType.OPTIONAL, value=most_common))
                    continue
        
        # 正常位置：使用混合策略
        element = create_element_with_mixed_strategy(amino_acids, total_words, strict_mode)
        elements.append(element)
    
    # 确保规则中至少有一定比例的性质类
    elements = ensure_property_class_ratio(elements)
    
    rule = GeneralizationRule(
        elements=elements,
        covered_words=word_originals,
        covered_count=len(word_originals)
    )
    
    return rule


def create_element_with_mixed_strategy(amino_acids: Set[str], 
                                       word_count: int,
                                       strict_mode: bool = False) -> RuleElement:
    """
    使用混合策略创建规则元素
    核心目标：最大化使用性质类，同时保持规则的精细度
    策略优先级：
    1. 单个氨基酸：强制使用Tier1/Tier2性质类
    2. 多个氨基酸：优先尝试精确匹配的性质类
    3. 如果无法精确匹配，尝试找到覆盖最多的性质类（允许部分匹配）
    4. 只有在完全无法找到性质类时，才使用选择类
    """
    if len(amino_acids) == 1:
        aa = list(amino_acids)[0]
        prop = find_best_property_for_single_amino(aa, avoid_tier3=True)
        if prop:
            return create_property_element(prop)
        return create_exact_element(aa)
    
    # 策略1：优先尝试Tier1性质类（精确匹配）
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 1 and amino_acids.issubset(prop_amino_acids):
            return create_property_element(prop_name)
    
    # 策略2：尝试Tier1性质类（部分匹配，至少50%）
    best_tier1_match = None
    best_tier1_ratio = 0
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 1:
            overlap = amino_acids & prop_amino_acids
            ratio = len(overlap) / len(amino_acids)
            if ratio > best_tier1_ratio:
                best_tier1_ratio = ratio
                best_tier1_match = prop_name
    if best_tier1_match and best_tier1_ratio >= 0.5:
        return create_property_element(best_tier1_match)
    
    # 策略3：尝试Tier2性质类（精确匹配）
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 2 and amino_acids.issubset(prop_amino_acids):
            return create_property_element(prop_name)
    
    # 策略4：尝试Tier2性质类（部分匹配，至少40%）
    best_tier2_match = None
    best_tier2_ratio = 0
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 2:
            overlap = amino_acids & prop_amino_acids
            ratio = len(overlap) / len(amino_acids)
            if ratio > best_tier2_ratio:
                best_tier2_ratio = ratio
                best_tier2_match = prop_name
    if best_tier2_match and best_tier2_ratio >= 0.4:
        return create_property_element(best_tier2_match)
    
    # 策略5：对于2-3个氨基酸，尝试找到共享的性质类
    if 2 <= len(amino_acids) <= 3:
        shared_props = find_shared_property(amino_acids)
        if shared_props:
            # 优先选择Tier1/Tier2性质类
            for prop in shared_props:
                for prop_name, _, tier in TIERED_SEMANTIC_GROUPS:
                    if prop_name == prop and tier <= 2:
                        return create_property_element(prop)
            return create_property_element(shared_props[0])
    
    # 策略6：尝试组合性质类
    combined_prop = find_combined_property(amino_acids)
    if combined_prop:
        return create_property_element(combined_prop)
    
    # 策略7：尝试多性质类组合
    multi_prop = find_multi_property_combination(amino_acids)
    if multi_prop:
        return create_property_element(multi_prop)
    
    # 策略8：对于4个氨基酸，如果它们都属于同一个性质类，使用性质类
    if len(amino_acids) == 4:
        for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
            if amino_acids.issubset(prop_amino_acids):
                return create_property_element(prop_name)
    
    # 策略9：找到覆盖最多的性质类（至少30%匹配，且不是polar/nonpolar）
    best_prop = find_best_property_for_amino_acids(amino_acids)
    if best_prop and best_prop not in ['polar', 'nonpolar']:
        prop_aas = [aas for n, aas, t in TIERED_SEMANTIC_GROUPS if n == best_prop][0]
        overlap = amino_acids & prop_aas
        if len(overlap) >= len(amino_acids) * 0.3:
            return create_property_element(best_prop)
    
    # 策略10：尝试找到能覆盖所有氨基酸的性质类（包括Tier3，但排除polar/nonpolar）
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if amino_acids.issubset(prop_amino_acids) and prop_name not in ['polar', 'nonpolar']:
            return create_property_element(prop_name)
    
    # 策略11：对于5个以上氨基酸，如果大部分属于同一个Tier3性质类（排除polar/nonpolar），使用该性质类
    if len(amino_acids) >= 5:
        for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
            if tier == 3 and prop_name not in ['polar', 'nonpolar']:
                overlap = amino_acids & prop_amino_acids
                if len(overlap) >= len(amino_acids) * 0.6:
                    return create_property_element(prop_name)
    
    # 最后回退到选择类（最多显示4个氨基酸）
    sorted_aas = sorted(amino_acids)[:4]
    return create_choice_element(''.join(sorted_aas))


def find_multi_property_combination(amino_acids: Set[str]) -> Optional[str]:
    """
    尝试找到多个Tier1/Tier2性质类的组合来覆盖所有氨基酸
    返回组合后的性质类名称，如 "acidic_or_basic"
    """
    # 获取所有Tier1/Tier2性质类
    tier1_props = [(name, aas) for name, aas, tier in TIERED_SEMANTIC_GROUPS if tier == 1]
    tier2_props = [(name, aas) for name, aas, tier in TIERED_SEMANTIC_GROUPS if tier == 2]
    all_props = tier1_props + tier2_props
    
    # 尝试2个性质类的组合
    for i, (prop1_name, prop1_aas) in enumerate(all_props):
        group1 = amino_acids & prop1_aas
        if len(group1) > 0:
            remaining = amino_acids - group1
            if len(remaining) == 0:
                return prop1_name
            
            for j, (prop2_name, prop2_aas) in enumerate(all_props):
                if i != j:
                    group2 = remaining & prop2_aas
                    if len(group1) + len(group2) == len(amino_acids):
                        # 确保是不同的性质类
                        if prop1_name != prop2_name:
                            return f"{prop1_name}_or_{prop2_name}"
    
    # 尝试3个性质类的组合
    for i, (prop1_name, prop1_aas) in enumerate(all_props):
        group1 = amino_acids & prop1_aas
        if len(group1) > 0:
            remaining1 = amino_acids - group1
            if len(remaining1) == 0:
                return prop1_name
            
            for j, (prop2_name, prop2_aas) in enumerate(all_props):
                if i != j:
                    group2 = remaining1 & prop2_aas
                    if len(group1) + len(group2) > 0:
                        remaining2 = remaining1 - group2
                        if len(group1) + len(group2) == len(amino_acids):
                            if prop1_name != prop2_name:
                                return f"{prop1_name}_or_{prop2_name}"
                        
                        for k, (prop3_name, prop3_aas) in enumerate(all_props):
                            if k != i and k != j:
                                group3 = remaining2 & prop3_aas
                                if len(group1) + len(group2) + len(group3) == len(amino_acids):
                                    if len(set([prop1_name, prop2_name, prop3_name])) >= 2:
                                        return f"{prop1_name}_or_{prop2_name}_or_{prop3_name}"
    
    return None


def find_best_property_for_single_amino(aa: str, avoid_tier3: bool = False) -> Optional[str]:
    """
    为单个氨基酸找到最合适的性质类
    优先选择Tier1，其次Tier2
    如果avoid_tier3=True，则不使用Tier3性质类
    """
    # 优先返回Tier1性质类
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 1 and aa in prop_amino_acids:
            return prop_name
    
    # 返回Tier2性质类
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 2 and aa in prop_amino_acids:
            return prop_name
    
    # 如果不避免Tier3，返回Tier3性质类
    if not avoid_tier3:
        for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
            if tier == 3 and aa in prop_amino_acids:
                return prop_name
    
    return None


def find_combined_property(amino_acids: Set[str]) -> Optional[str]:
    """
    尝试将氨基酸分成两个组，每组对应一个性质类
    返回组合后的性质类名称
    """
    # 尝试Tier1 + Tier1组合
    tier1_props = [(name, aas) for name, aas, tier in TIERED_SEMANTIC_GROUPS if tier == 1]
    for i, (prop1_name, prop1_aas) in enumerate(tier1_props):
        group1 = amino_acids & prop1_aas
        if len(group1) > 0:
            for j, (prop2_name, prop2_aas) in enumerate(tier1_props):
                if i != j:
                    group2 = amino_acids & prop2_aas
                    if len(group1) + len(group2) == len(amino_acids):
                        return f"{prop1_name}_or_{prop2_name}"
    
    # 尝试Tier1 + Tier2组合
    tier2_props = [(name, aas) for name, aas, tier in TIERED_SEMANTIC_GROUPS if tier == 2]
    for prop1_name, prop1_aas in tier1_props:
        group1 = amino_acids & prop1_aas
        if len(group1) > 0:
            for prop2_name, prop2_aas in tier2_props:
                group2 = amino_acids & prop2_aas
                if len(group1) + len(group2) == len(amino_acids):
                    return f"{prop1_name}_or_{prop2_name}"
    
    return None


def find_shared_property(amino_acids: Set[str]) -> List[str]:
    """
    找到一组氨基酸共享的所有性质类
    """
    if not amino_acids:
        return []
    
    # 获取第一个氨基酸的性质类
    first_aa = list(amino_acids)[0]
    shared_props = set()
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if first_aa in prop_amino_acids:
            shared_props.add(prop_name)
    
    # 检查其他氨基酸是否也有这些性质类
    for aa in amino_acids:
        aa_props = set()
        for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
            if aa in prop_amino_acids:
                aa_props.add(prop_name)
        shared_props = shared_props & aa_props
    
    # 按Tier排序
    result = []
    for prop_name, _, tier in TIERED_SEMANTIC_GROUPS:
        if prop_name in shared_props:
            result.append(prop_name)
    
    return result


def ensure_property_class_ratio(elements: List[RuleElement], min_ratio: float = 0.5) -> List[RuleElement]:
    """
    确保规则中至少包含一定比例的性质类（最低50%）
    如果性质类比例不足，尝试将一些choice/exact转换为性质类
    """
    if not elements:
        return elements
    
    property_count = sum(1 for e in elements if e.type == RuleElementType.PROPERTY)
    total_count = len(elements)
    
    if property_count / total_count >= min_ratio:
        return elements
    
    # 需要转换一些元素
    needed = max(1, int(min_ratio * total_count) - property_count)
    converted = 0
    
    for i, elem in enumerate(elements):
        if converted >= needed:
            break
        
        if elem.type == RuleElementType.CHOICE:
            # 尝试将choice转换为性质类
            aas = set(elem.value)
            # 首先尝试找覆盖最多的性质类
            best_prop = find_best_property_for_amino_acids(aas)
            if best_prop and best_prop not in ['polar', 'nonpolar']:
                elements[i] = create_property_element(best_prop)
                converted += 1
                continue
            
            # 尝试找共享性质类
            shared_props = find_shared_property(aas)
            if shared_props:
                # 优先选择Tier1/Tier2性质类
                for prop in shared_props:
                    for prop_name, _, tier in TIERED_SEMANTIC_GROUPS:
                        if prop_name == prop and tier <= 2:
                            elements[i] = create_property_element(prop)
                            converted += 1
                            break
                    if converted > 0:
                        break
        elif elem.type == RuleElementType.EXACT:
            # 尝试将exact转换为性质类
            prop = find_best_property_for_single_amino(elem.value, avoid_tier3=True)
            if prop:
                elements[i] = create_property_element(prop)
                converted += 1
    
    return elements


def create_element_for_position(amino_acids: Set[str], 
                               word_count: int,
                               strict_mode: bool = False) -> RuleElement:
    """
    根据位置上的氨基酸分布创建规则元素
    策略：
    1. 单个氨基酸：优先使用Tier1性质类
    2. 多个氨基酸：优先使用最精细的性质类（Tier1 > Tier2 > Tier3）
    3. 如果不完全匹配，使用最接近的性质类
    4. strict_mode下：使用更精确的匹配
    """
    if len(amino_acids) == 1:
        aa = list(amino_acids)[0]
        if strict_mode:
            return create_exact_element(aa)
        prop = find_property_for_single_amino(aa)
        if prop:
            return create_property_element(prop)
        return create_exact_element(aa)
    
    # 策略1：优先尝试Tier1性质类（精确匹配）
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 1 and amino_acids.issubset(prop_amino_acids):
            return create_property_element(prop_name)
    
    # 策略2：尝试Tier1性质类（70%以上匹配）
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 1:
            overlap = amino_acids & prop_amino_acids
            if len(overlap) >= len(amino_acids) * 0.7:
                return create_property_element(prop_name)
    
    # 策略3：尝试Tier2性质类（精确匹配）
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 2 and amino_acids.issubset(prop_amino_acids):
            return create_property_element(prop_name)
    
    # 策略4：尝试Tier2性质类（60%以上匹配）
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 2:
            overlap = amino_acids & prop_amino_acids
            if len(overlap) >= len(amino_acids) * 0.6:
                return create_property_element(prop_name)
    
    # 策略5：2个氨基酸优先尝试性质类
    if len(amino_acids) == 2:
        for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
            if amino_acids.issubset(prop_amino_acids):
                return create_property_element(prop_name)
    
    # 策略6：2-3个氨基酸用choice
    if len(amino_acids) <= 3:
        return create_choice_element(''.join(sorted(amino_acids)))
    
    # 策略7：严格模式下，使用choice
    if strict_mode:
        sorted_aas = sorted(amino_acids)[:5]
        return create_choice_element(''.join(sorted_aas))
    
    # 策略8：非严格模式下，尝试Tier3性质类（精确匹配）
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 3 and amino_acids.issubset(prop_amino_acids):
            return create_property_element(prop_name)
    
    # 策略9：非严格模式下，尝试Tier3性质类（50%以上匹配）
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 3:
            overlap = amino_acids & prop_amino_acids
            if len(overlap) >= len(amino_acids) * 0.5:
                return create_property_element(prop_name)
    
    # 策略10：对于4个以上氨基酸，强制使用最接近的性质类
    best_prop = find_best_property_for_amino_acids(amino_acids)
    if best_prop:
        return create_property_element(best_prop)
    
    # 最后回退到choice（最多显示5个氨基酸）
    sorted_aas = sorted(amino_acids)[:5]
    return create_choice_element(''.join(sorted_aas))


def find_best_property_for_amino_acids(amino_acids: Set[str]) -> Optional[str]:
    """
    为一组氨基酸找到最佳的性质类
    选择覆盖最多氨基酸的性质类
    """
    best_prop = None
    best_overlap = 0
    best_tier = 99
    
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        overlap = len(amino_acids & prop_amino_acids)
        # 优先选择Tier低的，其次选择覆盖度高的
        if (overlap > best_overlap) or (overlap == best_overlap and tier < best_tier):
            best_overlap = overlap
            best_prop = prop_name
            best_tier = tier
    
    # 至少覆盖2个氨基酸才使用性质类
    if best_prop and best_overlap >= 2:
        return best_prop
    return None


def find_property_for_single_amino(aa: str) -> Optional[str]:
    """
    为单个氨基酸找到最合适的性质类（优先Tier1）
    """
    # 优先返回Tier1性质类
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 1 and aa in prop_amino_acids:
            return prop_name
    # 返回Tier2性质类
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 2 and aa in prop_amino_acids:
            return prop_name
    # 返回Tier3性质类
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        if tier == 3 and aa in prop_amino_acids:
            return prop_name
    return None


def generate_cluster_based_rules(words: List[ProcessedWord]) -> List[GeneralizationRule]:
    """
    基于聚类的规则生成
    """
    rules = []
    
    # 按相似度分组
    groups = group_words_by_similarity(words)
    
    for group in groups:
        # 多序列对齐
        aligned = align_word_group(group)
        
        # 生成规则
        word_originals = {w.original for w in group}
        rule = build_rule_from_alignment(aligned, word_originals)
        
        if rule.elements:
            rules.append(rule)
    
    return rules


def generate_length_based_rules(words: List[ProcessedWord]) -> List[GeneralizationRule]:
    """
    按长度分组生成规则
    """
    rules = []
    
    length_groups = defaultdict(list)
    for word in words:
        length_groups[word.length].append(word)
    
    for length, group in length_groups.items():
        if len(group) < 2:
            continue
        
        aligned = align_word_group(group)
        word_originals = {w.original for w in group}
        rule = build_rule_from_alignment(aligned, word_originals)
        
        if rule.elements:
            rules.append(rule)
    
    return rules


def generate_original_word_rules(uncovered_words: Set[str]) -> List[GeneralizationRule]:
    """
    为未覆盖的词生成原始规则
    每个词生成完整的规则，包含所有氨基酸的确切匹配
    """
    rules = []
    
    for word in uncovered_words:
        elements = []
        parts = word.split('_')
        
        for i, part in enumerate(parts):
            if part == '':
                elements.append(create_wildcard_any_element())
            else:
                for aa in part:
                    elements.append(create_exact_element(aa))
        
        rule = GeneralizationRule(
            elements=elements,
            covered_words={word},
            covered_count=1
        )
        
        rules.append(rule)
    
    return rules


def generate_rules_for_fragment(words: List[str],
                                word_scores: Dict[str, float],
                                max_time: float = 60.0) -> List[GeneralizationRule]:
    """
    为单个fragment生成泛化规则
    核心策略：聚类 + MSA + 属性类泛化
    """
    processed_words = [preprocess_word(w) for w in words]
    
    # 第一层：基于聚类的规则
    cluster_rules = generate_cluster_based_rules(processed_words)
    
    # 第二层：基于长度的规则（补充覆盖）
    length_rules = generate_length_based_rules(processed_words)
    
    all_rules = cluster_rules + length_rules
    
    # 计算当前覆盖情况
    covered_words = set()
    for rule in all_rules:
        covered_words.update(rule.covered_words)
    
    # 第三层：回退到原始词规则
    uncovered_words = set(words) - covered_words
    original_rules = generate_original_word_rules(uncovered_words)
    
    all_rules += original_rules
    
    # 添加score信息
    for rule in all_rules:
        scores = [word_scores.get(word, 0.0) for word in rule.covered_words]
        rule.average_score = sum(scores) / len(scores) if scores else 0.0
    
    return all_rules