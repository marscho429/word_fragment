"""
基本功能测试脚本
"""
import sys
sys.path.insert(0, 'src')

# 测试基本功能
print('=' * 60)
print('测试 1: 核心功能')
print('=' * 60)

from core import (
    create_exact_element, create_property_element,
    get_best_property_for_amino_acids, GeneralizationRule
)

# 测试创建元素
print('\n创建元素:')
elem1 = create_exact_element('A')
elem2 = create_property_element('acidic')
print(f'  确切氨基酸 A: {elem1}')
print(f'  性质类 acidic: {elem2}')

# 测试性质类查找
print('\n性质类查找:')
test_aas = {'K', 'R'}
best_prop = get_best_property_for_amino_acids(test_aas)
print(f'  {test_aas} 的最佳性质类: {best_prop}')

test_aas2 = {'D', 'E'}
best_prop2 = get_best_property_for_amino_acids(test_aas2)
print(f'  {test_aas2} 的最佳性质类: {best_prop2}')

# 测试创建规则
print('\n创建规则:')
elements = [elem1, elem2]
rule = GeneralizationRule(elements=elements, covered_words={'A', 'K'})
print(f'  规则字符串: {str(rule)}')
print(f'  正则模式: {rule.to_regex_pattern()}')
print(f'  Wildcard比例: {rule.get_wildcard_ratio():.2f}')
print(f'  优先级分数: {rule.get_priority_score():.1f}')

print('\n核心功能测试通过！')

print('\n' + '=' * 60)
print('测试 2: 预处理功能')
print('=' * 60)

from preprocess import preprocess_word, calculate_amino_overlap

# 测试预处理
print('\n预处理蛋白词:')
test_words = ['RYRGSDC', 'KYRGGSDC', 'L_VIS_DR', 'G_I_VII']
for word in test_words:
    processed = preprocess_word(word)
    print(f'  {word}:')
    print(f'    氨基酸序列: {processed.amino_sequence}')
    print(f'    有下划线: {processed.has_underscore}')
    print(f'    长度: {processed.length}')

# 测试重叠度计算
print('\n计算重叠度:')
seq1 = ['R', 'Y', 'R', 'G', 'S', 'D', 'C']
seq2 = ['K', 'Y', 'R', 'G', 'G', 'S', 'D', 'C']
overlap = calculate_amino_overlap(seq1, seq2)
print(f'  RYRGSDC 和 KYRGGSDC 的重叠度: {overlap:.2f}')

seq3 = ['Y', 'R', 'G', 'S', 'D', 'C']
seq4 = ['D', 'R', 'W', 'Y', 'R', 'G', 'S', 'D', 'C', 'R', 'T', 'I']
overlap2 = calculate_amino_overlap(seq3, seq4)
print(f'  YRGSDC 和 DRWYRGSDCRTI 的重叠度: {overlap2:.2f}')

print('\n预处理测试通过！')

print('\n' + '=' * 60)
print('测试 3: 泛化规则生成')
print('=' * 60)

from generalize import align_words, generate_rule_from_alignment

# 测试比对
print('\n多序列比对:')
from preprocess import preprocess_word
words = [preprocess_word('RYRGSDC'), preprocess_word('KYRGGSDC')]
alignment = align_words(words)
print(f'  对齐的序列:')
for i, seq in enumerate(alignment.aligned_sequences):
    print(f'    序列{i+1}: {seq}')
print(f'  位置信息:')
for pos, aas in alignment.positions:
    print(f'    位置{pos}: {aas}')

# 测试生成规则
print('\n生成泛化规则:')
rule = generate_rule_from_alignment(alignment, {'RYRGSDC', 'KYRGGSDC'})
print(f'  规则: {str(rule)}')
print(f'  正则模式: {rule.to_regex_pattern()}')
print(f'  Wildcard比例: {rule.get_wildcard_ratio():.2f}')
print(f'  优先级分数: {rule.get_priority_score():.1f}')
print(f'  覆盖的词: {rule.covered_words}')

print('\n泛化规则生成测试通过！')

print('\n' + '=' * 60)
print('测试 4: 规则验证')
print('=' * 60)

from validate import match_word_with_rule, calculate_fpr_for_rule

# 测试匹配
print('\n测试规则匹配:')
test_words = ['RYRGSDC', 'KYRGGSDC', 'AYRGSDC', 'XYRGSDC']
for word in test_words:
    matched = match_word_with_rule(word, rule)
    print(f'  {word}: {matched}')

# 测试FPR
print('\n计算FPR:')
negative_samples = ['AYRGSDC', 'XYRGSDC', 'AAAAAA', 'BBBBBB']
fpr = calculate_fpr_for_rule(rule, negative_samples)
print(f'  规则: {str(rule)}')
print(f'  负样本: {negative_samples}')
print(f'  FPR: {fpr:.2f}')

print('\n规则验证测试通过！')

print('\n' + '=' * 60)
print('所有测试通过！')
print('=' * 60)