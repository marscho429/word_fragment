"""
测试前5个片段，验证覆盖率>=85%，输出JSON结果
"""
import sys
sys.path.insert(0, 'src')

import pickle
import json
import time
import random
from typing import List, Dict, Set

from core import RuleElementType
from generalize import generate_rules_for_fragment
from validate import (
    calculate_fpr_for_rule_set, match_word_with_rule,
    compute_coverage_metrics, validate_rule_set
)
from preprocess import preprocess_word

def load_data():
    """加载数据"""
    with open('dataset/origin.pkl', 'rb') as f:
        frag_to_words = pickle.load(f)
    
    with open('dataset/pwfm_hard_1024_1024_processed_ww2_wordfrag2score.pkl', 'rb') as f:
        wordfrag2score = pickle.load(f)
    
    return frag_to_words, wordfrag2score

def get_all_words(wordfrag2score: Dict) -> Set[str]:
    """获取所有蛋白词"""
    all_words = set()
    for key in wordfrag2score.keys():
        parts = key.split()
        if len(parts) >= 1:
            all_words.add(parts[0])
    return all_words

def process_fragment(fragment_id: str, 
                     words: List[str], 
                     wordfrag2score: Dict,
                     all_words: Set[str],
                     sample_size: int = 3000) -> Dict:
    """处理单个fragment"""
    start_time = time.time()
    
    word_set = set(words)
    
    # 获取score
    word_scores = {}
    for word in words:
        key = f"{word} {fragment_id}"
        word_scores[word] = wordfrag2score.get(key, 0.0)
    
    # 生成规则
    rules = generate_rules_for_fragment(words, word_scores)
    
    # 计算覆盖率指标
    coverage_metrics = compute_coverage_metrics(rules, word_set)
    
    # 生成负样本（优化：预计算）
    negative_words = all_words - word_set
    if len(negative_words) > sample_size:
        negative_sample = random.sample(list(negative_words), sample_size)
    else:
        negative_sample = list(negative_words)
    
    # 计算FPR（只对前50条规则计算）
    rules_for_fpr = rules[:50]
    fpr = calculate_fpr_for_rule_set(rules_for_fpr, negative_sample)
    
    # 验证规则是否匹配原始词（只验证前50条）
    all_match = True
    for rule in rules[:50]:
        for word in rule.covered_words:
            if not match_word_with_rule(word, rule):
                all_match = False
                break
        if not all_match:
            break
    
    # 统计规则类型
    multi_word_rules = [r for r in rules if r.covered_count >= 2]
    single_word_rules = [r for r in rules if r.covered_count == 1]
    
    # 统计包含性质类的规则数量
    prop_rules_count = sum(1 for r in rules if any(e.type == RuleElementType.PROPERTY for e in r.elements))
    
    # 转换规则为字典格式
    rules_dict = []
    for rule in rules[:30]:
        rule_dict = {
            'Pattern': str(rule),
            'Covered_Words': sorted(list(rule.covered_words)),
            'Covered_Count': rule.covered_count,
            'Average_Score': rule.average_score,
            'Wildcard_Ratio': rule.get_wildcard_ratio(),
            'Priority_Score': rule.get_priority_score(),
            'Has_Property': any(e.type == 'property' for e in rule.elements)
        }
        rules_dict.append(rule_dict)
    
    processing_time = time.time() - start_time
    
    result = {
        'fragment_id': fragment_id,
        'total_words': len(words),
        'coverage_ratio': coverage_metrics['coverage_ratio'],
        'fpr': fpr,
        'all_match': all_match,
        'rule_count': len(rules),
        'multi_word_rule_count': len(multi_word_rules),
        'single_word_rule_count': len(single_word_rules),
        'property_rule_count': prop_rules_count,
        'property_rule_ratio': prop_rules_count / len(rules) if rules else 0,
        'avg_wildcard_ratio': sum(r.get_wildcard_ratio() for r in rules) / len(rules) if rules else 0,
        'avg_priority_score': sum(r.get_priority_score() for r in rules) / len(rules) if rules else 0,
        'processing_time': processing_time,
        'rules': rules_dict,
        'rules_truncated': len(rules) > 30
    }
    
    return result

def main():
    print('=' * 70)
    print('测试前5个片段 - 泛化规则生成')
    print('=' * 70)
    
    # 加载数据
    print('加载数据...')
    frag_to_words, wordfrag2score = load_data()
    all_words = get_all_words(wordfrag2score)
    print(f'总fragments: {len(frag_to_words)}')
    print(f'总蛋白词: {len(all_words)}')
    
    # 选择前5个片段
    top_fragments = list(frag_to_words.keys())[:5]
    print(f'\n测试片段: {top_fragments}')
    
    # 处理每个片段
    results = []
    total_coverage = 0.0
    total_fpr = 0.0
    total_rules = 0
    
    for i, fragment_id in enumerate(top_fragments):
        print(f'\n{"-" * 70}')
        print(f'处理片段 {i+1}/5: {fragment_id}')
        words = list(frag_to_words[fragment_id])
        print(f'词数量: {len(words)}')
        
        result = process_fragment(fragment_id, words, wordfrag2score, all_words)
        
        print(f'覆盖率: {result["coverage_ratio"]:.2%}')
        print(f'FPR: {result["fpr"]:.4f}')
        print(f'规则数: {result["rule_count"]} (泛化规则: {result["multi_word_rule_count"]}, 原始规则: {result["single_word_rule_count"]})')
        print(f'处理时间: {result["processing_time"]:.2f}s')
        
        results.append(result)
        total_coverage += result['coverage_ratio']
        total_fpr += result['fpr']
        total_rules += result['rule_count']
        
        # 打印部分规则示例
        print('\n部分规则示例:')
        for j, rule in enumerate(result['rules'][:5]):
            print(f'  {j+1}. {rule["Pattern"]}')
            print(f'     覆盖: {rule["Covered_Count"]}个词, Score: {rule["Average_Score"]:.3f}')
    
    # 统计汇总
    print(f'\n{"=" * 70}')
    print('测试结果汇总')
    print('=' * 70)
    print(f'测试片段数: {len(results)}')
    print(f'平均覆盖率: {total_coverage / len(results):.2%}')
    print(f'平均FPR: {total_fpr / len(results):.4f}')
    print(f'总规则数: {total_rules}')
    
    # 检查是否达到目标
    target_met = all(r['coverage_ratio'] >= 0.85 for r in results)
    print(f'\n覆盖率目标(>=85%): {"✓ 达成" if target_met else "✗ 未达成"}')
    
    # 输出JSON结果
    output_data = {
        'summary': {
            'total_fragments': len(results),
            'average_coverage': total_coverage / len(results),
            'average_fpr': total_fpr / len(results),
            'total_rules': total_rules,
            'target_met': target_met
        },
        'fragments': results
    }
    
    with open('top5_results.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f'\n结果已保存到: top5_results.json')
    print('=' * 70)

if __name__ == '__main__':
    main()