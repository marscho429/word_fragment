"""
核心数据结构和常量定义
"""
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
import re


# 氨基酸性质分类（根据README中的TIERED_SEMANTIC_GROUPS）
TIERED_SEMANTIC_GROUPS = [
    # Tier 1: 绝对高优组 (极度特异，2~3个元素)
    ('acidic', {'D', 'E'}, 1),
    ('basic', {'K', 'R', 'H'}, 1),
    ('sulfur', {'C', 'M'}, 1),
    ('amide', {'N', 'Q'}, 1),
    ('hydroxyl', {'S', 'T', 'Y'}, 1),
    ('tiny', {'G', 'A', 'S'}, 1),
    ('aromatic', {'F', 'Y', 'W'}, 2),
    
    # Tier 2: 中优先级 (特定骨架特征，4~6个元素)
    ('aliphatic', {'A', 'V', 'L', 'I'}, 2),
    ('cyclic', {'P', 'F', 'Y', 'W', 'H'}, 2),
    ('uncharged_polar', {'S', 'T', 'N', 'Q', 'Y', 'C'}, 2),
    
    # Tier 3: 兜底低优组 (宽泛分类，8~11个元素)
    ('small', {'G', 'A', 'S', 'C', 'D', 'P', 'N', 'T'}, 3),
    ('nonpolar', {'A', 'V', 'L', 'I', 'P', 'F', 'M', 'W', 'G'}, 3),
    ('polar', {'S', 'T', 'N', 'Q', 'Y', 'C', 'D', 'E', 'K', 'R', 'H'}, 3),
    ('bulky', {'F', 'Y', 'W', 'R', 'K', 'E', 'Q', 'M', 'L', 'I'}, 3)
]

# 构建氨基酸到性质类的映射字典
AMINO_ACID_TO_PROPERTIES: Dict[str, List[Tuple[str, int]]] = {}
for prop_name, amino_acids, tier in TIERED_SEMANTIC_GROUPS:
    for aa in amino_acids:
        if aa not in AMINO_ACID_TO_PROPERTIES:
            AMINO_ACID_TO_PROPERTIES[aa] = []
        AMINO_ACID_TO_PROPERTIES[aa].append((prop_name, tier))

# 所有标准氨基酸
ALL_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')


class RuleElementType(Enum):
    """规则元素类型"""
    EXACT = "exact"           # 确切氨基酸，如 A
    PROPERTY = "property"     # 性质类，如 [acidic]
    OPTIONAL = "optional"     # 可选类，如 (A)
    WILDCARD_FIXED = "wildcard_fixed"     # 固定数量的任意氨基酸，如 x(3)
    WILDCARD_RANGE = "wildcard_range"     # 范围数量的任意氨基酸，如 x(1,3)
    WILDCARD_ANY = "wildcard_any"         # 一个或多个任意氨基酸，如 _
    CHOICE = "choice"         # 多选一，如 [ABC]


@dataclass
class RuleElement:
    """单个规则元素"""
    type: RuleElementType
    value: str  # 具体值，如 'A', 'acidic', '1,3'等
    min_count: int = 1  # 最小数量（用于wildcard）
    max_count: int = 1  # 最大数量（用于wildcard）
    
    def __str__(self):
        """转换为字符串表示"""
        if self.type == RuleElementType.EXACT:
            return self.value
        elif self.type == RuleElementType.PROPERTY:
            return f"[{self.value}]"
        elif self.type == RuleElementType.OPTIONAL:
            return f"({self.value})"
        elif self.type == RuleElementType.WILDCARD_FIXED:
            return f"x({self.min_count})"
        elif self.type == RuleElementType.WILDCARD_RANGE:
            return f"x({self.min_count},{self.max_count})"
        elif self.type == RuleElementType.WILDCARD_ANY:
            return "_"
        elif self.type == RuleElementType.CHOICE:
            return "{" + ",".join(sorted(self.value)) + "}"
        return ""
    
    def to_pattern(self) -> str:
        """转换为正则表达式模式"""
        if self.type == RuleElementType.EXACT:
            return self.value
        elif self.type == RuleElementType.PROPERTY:
            for prop_name, amino_acids, tier in TIERED_SEMANTIC_GROUPS:
                if prop_name == self.value:
                    return f"[{''.join(amino_acids)}]"
            return self.value
        elif self.type == RuleElementType.OPTIONAL:
            return f"({self.value})?"
        elif self.type == RuleElementType.WILDCARD_FIXED:
            return f"[ACDEFGHIKLMNPQRSTVWY]{{{self.min_count}}}"
        elif self.type == RuleElementType.WILDCARD_RANGE:
            return f"[ACDEFGHIKLMNPQRSTVWY]{{{self.min_count},{self.max_count}}}"
        elif self.type == RuleElementType.WILDCARD_ANY:
            return f"[ACDEFGHIKLMNPQRSTVWY]+"
        elif self.type == RuleElementType.CHOICE:
            return f"[{self.value}]"
        return ""


@dataclass
class GeneralizationRule:
    """泛化规则"""
    elements: List[RuleElement]
    covered_words: Set[str] = field(default_factory=set)
    covered_count: int = 0
    average_score: float = 0.0
    
    def __str__(self):
        """转换为字符串表示"""
        return ''.join(str(e) for e in self.elements)
    
    def to_regex_pattern(self) -> str:
        """转换为正则表达式"""
        return ''.join(e.to_pattern() for e in self.elements)
    
    def get_wildcard_ratio(self) -> float:
        """计算任意氨基酸的比例"""
        total_length = len(self.elements)
        if total_length == 0:
            return 0.0
        
        wildcard_count = 0
        for elem in self.elements:
            if elem.type in [RuleElementType.WILDCARD_FIXED, 
                           RuleElementType.WILDCARD_RANGE, 
                           RuleElementType.WILDCARD_ANY]:
                # 任意氨基酸算作长度贡献
                avg_len = (elem.min_count + elem.max_count) / 2
                wildcard_count += avg_len
        
        # 简化计算：任意氨基酸类型的元素数量占比
        wildcard_elements = sum(1 for e in self.elements 
                               if e.type in [RuleElementType.WILDCARD_FIXED,
                                           RuleElementType.WILDCARD_RANGE,
                                           RuleElementType.WILDCARD_ANY])
        return wildcard_elements / total_length if total_length > 0 else 0.0
    
    def get_priority_score(self) -> float:
        """
        计算规则的优先级分数（越高越好）
        优先级：确切氨基酸 >> 性质类 >> 可选类 > 任意氨基酸
        """
        score = 0.0
        for elem in self.elements:
            if elem.type == RuleElementType.EXACT:
                score += 10.0
            elif elem.type == RuleElementType.PROPERTY:
                # Tier 1性质类分数更高
                for prop_name, amino_acids, tier in TIERED_SEMANTIC_GROUPS:
                    if prop_name == elem.value:
                        score += 8.0 - tier  # Tier 1: 7分，Tier 2: 6分，Tier 3: 5分
                        break
            elif elem.type == RuleElementType.CHOICE:
                score += 7.0
            elif elem.type == RuleElementType.OPTIONAL:
                score += 5.0
            elif elem.type == RuleElementType.WILDCARD_FIXED:
                score += 3.0 / elem.min_count  # 数量越少越好
            elif elem.type == RuleElementType.WILDCARD_RANGE:
                avg = (elem.min_count + elem.max_count) / 2
                score += 2.0 / avg if avg > 0 else 0.0
            elif elem.type == RuleElementType.WILDCARD_ANY:
                score += 1.0
        
        return score


def get_best_property_for_amino_acids(amino_acids: Set[str]) -> Optional[str]:
    """
    为一组氨基酸找到最佳的性质类
    返回能覆盖所有给定氨基酸的最小性质类
    """
    # 按tier排序，优先选择Tier 1
    candidates = []
    
    for prop_name, prop_amino_acids, tier in TIERED_SEMANTIC_GROUPS:
        # 检查这个性质类是否包含所有给定的氨基酸
        if amino_acids.issubset(prop_amino_acids):
            candidates.append((prop_name, len(prop_amino_acids), tier))
    
    if not candidates:
        return None
    
    # 选择最优的：优先tier低，其次集合小
    candidates.sort(key=lambda x: (x[2], x[1]))
    return candidates[0][0]


def parse_word(word: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    解析蛋白词，分离氨基酸和下划线位置
    返回：(氨基酸列表, 下划线位置列表)
    下划线位置：(start_index, end_index) - 在氨基酸列表中的位置范围
    """
    amino_acids = []
    underscore_positions = []
    
    # 分割蛋白词，识别下划线
    parts = word.split('_')
    
    current_pos = 0
    for i, part in enumerate(parts):
        if part == '':
            # 这是下划线部分（连续的空字符串）
            if i > 0 and current_pos > 0:
                # 下划线在中间，记录位置
                underscore_positions.append((current_pos, current_pos + 1))
        else:
            # 这是氨基酸部分
            amino_acids.extend(list(part))
            current_pos += len(part)
    
    return amino_acids, underscore_positions


def is_valid_amino_acid(aa: str) -> bool:
    """检查是否是有效氨基酸"""
    return aa in ALL_AMINO_ACIDS


def create_exact_element(aa: str) -> RuleElement:
    """创建确切氨基酸元素"""
    return RuleElement(type=RuleElementType.EXACT, value=aa)


def create_property_element(prop_name: str) -> RuleElement:
    """创建性质类元素"""
    return RuleElement(type=RuleElementType.PROPERTY, value=prop_name)


def create_optional_element(aa: str) -> RuleElement:
    """创建可选元素"""
    return RuleElement(type=RuleElementType.OPTIONAL, value=aa)


def create_wildcard_fixed_element(n: int) -> RuleElement:
    """创建固定数量任意氨基酸元素"""
    return RuleElement(
        type=RuleElementType.WILDCARD_FIXED, 
        value=str(n),
        min_count=n,
        max_count=n
    )


def create_wildcard_range_element(min_n: int, max_n: int) -> RuleElement:
    """创建范围数量任意氨基酸元素"""
    return RuleElement(
        type=RuleElementType.WILDCARD_RANGE,
        value=f"{min_n},{max_n}",
        min_count=min_n,
        max_count=max_n
    )


def create_wildcard_any_element() -> RuleElement:
    """创建一个或多个任意氨基酸元素"""
    return RuleElement(
        type=RuleElementType.WILDCARD_ANY,
        value="_",
        min_count=1,
        max_count=999  # 最大值设为一个较大的数
    )


def create_choice_element(choices: str) -> RuleElement:
    """创建多选一元素"""
    return RuleElement(type=RuleElementType.CHOICE, value=choices)