"""
小规模实际数据测试
"""
import sys
sys.path.insert(0, 'src')

import pickle
import time

print('=' * 60)
print('测试实际数据处理（小规模）')
print('=' * 60)

# 加载一个fragment的数据进行测试
with open('dataset/origin.pkl', 'rb') as f:
    frag_to_words = pickle.load(f)

# 选择第一个fragment进行测试
test_frag = list(frag_to_words.keys())[0]
test_words = list(frag_to_words[test_frag])[:20]  # 只取20个词测试

print(f'\n测试Fragment: {test_frag}')
print(f'测试词数量: {len(test_words)}')
print(f'部分词示例:')
for i, word in enumerate(test_words[:5]):
    print(f'  {i+1}. {word}')

# 加载score数据
with open('dataset/pwfm_hard_1024_1024_processed_ww2_wordfrag2score.pkl', 'rb') as f:
    wordfrag2score = pickle.load(f)

word_scores = {}
for word in test_words:
    key = f"{word} {test_frag}"
    word_scores[word] = wordfrag2score.get(key, 0.0)

print(f'\n部分score示例:')
for word in test_words[:5]:
    print(f'  {word}: {word_scores[word]:.3f}')

# 测试规则生成
print('\n开始生成规则...')
from generalize import generate_rules_for_fragment
from preprocess import preprocess_word

start_time = time.time()

try:
    # 先预处理
    processed_words = [preprocess_word(w) for w in test_words]
    print(f'预处理完成，耗时: {time.time() - start_time:.2f}s')
    
    # 生成规则（限制时间）
    rules = generate_rules_for_fragment(
        test_words,
        word_scores,
        max_time=10.0  # 测试时限制为10秒
    )
    
    print(f'规则生成完成，耗时: {time.time() - start_time:.2f}s')
    print(f'生成规则数量: {len(rules)}')
    
    if rules:
        print(f'\n生成的规则示例:')
        for i, rule in enumerate(rules[:5]):
            print(f'  {i+1}. 规则: {str(rule)}')
            print(f'     覆盖词数: {rule.covered_count}')
            print(f'     平均score: {rule.average_score:.3f}')
            print(f'     Wildcard比例: {rule.get_wildcard_ratio():.2f}')
            print(f'     优先级: {rule.get_priority_score():.1f}')
            print(f'     部分覆盖词: {list(rule.covered_words)[:3]}')
    
    # 测试覆盖率
    covered_words = set()
    for rule in rules:
        covered_words.update(rule.covered_words)
    
    coverage = len(covered_words) / len(test_words) if test_words else 0
    print(f'\n覆盖率: {coverage:.2%} ({len(covered_words)}/{len(test_words)})')
    
    # 测试负样本FPR
    print('\n测试FPR...')
    all_words = set()
    with open('dataset/pwfm_hard_1024_1024_processed_ww2_wordfrag2score.pkl', 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            parts = key.split()
            if len(parts) >= 1:
                all_words.add(parts[0])
    
    negative_words = [w for w in all_words if w not in set(test_words)]
    
    from validate import calculate_fpr_for_rule_set
    # 随机采样1000个负样本
    import random
    if len(negative_words) > 1000:
        negative_sample = random.sample(negative_words, 1000)
    else:
        negative_sample = negative_words
    
    fpr = calculate_fpr_for_rule_set(rules, negative_sample)
    print(f'FPR (使用{len(negative_sample)}个负样本): {fpr:.4f}')
    
except Exception as e:
    print(f'错误: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('测试完成')
print('=' * 60)