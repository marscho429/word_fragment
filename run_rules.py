"""
规则生成器 - 凝聚式层次聚类 C++ 实现
仅负责加载数据和调用C++模块
"""
import sys
import pickle
import json
import time
import argparse
import os
import random


def load_origin_data(path='dataset/origin.pkl'):
    """加载原始数据"""
    print(f"Loading origin data from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    result = {}
    for frag_id, words in data.items():
        if isinstance(words, set):
            result[frag_id] = list(words)
        elif isinstance(words, dict):
            result[frag_id] = list(words.keys())
        else:
            result[frag_id] = list(words)

    print(f"  Loaded {len(result)} fragments")
    return result


def build_negative_pool(origin_data, current_frag_id, sample_size=1000):
    """从其他fragment中采样负样本"""
    pool = []
    for frag_id, words in origin_data.items():
        if frag_id != current_frag_id:
            pool.extend(words)
    if len(pool) > sample_size:
        random.seed(42)
        pool = random.sample(pool, sample_size)
    return pool


def main():
    parser = argparse.ArgumentParser(description='Generate rules using AHC C++ backend')
    parser.add_argument('--input', type=str, default='dataset/origin.pkl',
                       help='Input origin data')
    parser.add_argument('--output', type=str, default='outputs/rules_ahc.json',
                       help='Output results file')
    parser.add_argument('--num-fragments', type=int, default=None,
                       help='Number of fragments to process (None = all)')
    parser.add_argument('--target-coverage', type=float, default=0.85,
                       help='Target coverage rate')
    parser.add_argument('--max-fpr', type=float, default=0.1,
                       help='Maximum false positive rate')
    parser.add_argument('--negative-sample-size', type=int, default=1000,
                       help='Negative sample pool size for FPR calculation')

    args = parser.parse_args()

    # 导入C++模块
    sys.path.insert(0, 'src')
    import rule_generator_cpp as rg

    print("=" * 80)
    print("Rule Generation with AHC C++ Backend")
    print("=" * 80)

    # 加载数据
    origin_data = load_origin_data(args.input)

    # 获取要处理的fragment
    frag_ids = list(origin_data.keys())
    if args.num_fragments:
        frag_ids = frag_ids[:args.num_fragments]

    print(f"\nProcessing {len(frag_ids)} fragments...")

    # 全局负样本池（所有fragment共享，一次采样）
    print("Building global negative pool...")
    all_words = []
    for frag_id, words in origin_data.items():
        all_words.extend(words)
    print(f"  Total words across all fragments: {len(all_words)}")

    # 处理每个fragment
    results = {}
    total_stats = {
        'total_words': 0,
        'covered_words': 0,
        'gen_covered': 0,
        'total_rules': 0
    }

    start_time = time.time()

    for i, frag_id in enumerate(frag_ids):
        print(f"\n{'='*80}")
        print(f"Fragment {i+1}/{len(frag_ids)}: {frag_id}")
        print(f"{'='*80}")

        frag_start = time.time()
        words = origin_data[frag_id]

        # 构建该fragment的负样本池
        negative_pool = build_negative_pool(
            origin_data, frag_id, args.negative_sample_size
        )

        print(f"  Words: {len(words)}, Negative pool: {len(negative_pool)}")

        # 调用C++ AHC生成规则
        result = rg.generate_rules(
            words,
            negative_pool,
            args.target_coverage,
            args.max_fpr,
            1  # verbosity
        )

        # 构建输出
        rules = []
        for j, (pattern, score, covered_words, covered_count) in enumerate(zip(
            result['patterns'],
            result['scores'],
            result['covered_words_list'],
            result['covered_counts']
        )):
            rules.append({
                'Pattern': pattern,
                'Covered_Words': covered_words,
                'Covered_Count': covered_count,
                'Average_Score': score,
                'False_Positive_Rate': 0.0
            })

        frag_result = {
            'Fragment_ID': frag_id,
            'Total_Words': result['total_words'],
            'Rules': rules,
            'Coverage': result['coverage'],
            'Gen_Coverage': result.get('gen_coverage', 0.0),
            'Overall_FPR': result.get('fpr', 0.0)
        }

        results[frag_id] = frag_result

        frag_elapsed = time.time() - frag_start

        # 统计
        total_stats['total_words'] += result['total_words']
        total_stats['covered_words'] += result['total_covered']
        total_stats['gen_covered'] += result.get('gen_covered', 0)
        total_stats['total_rules'] += len(rules)

        # 打印摘要
        print(f"  Time: {frag_elapsed:.2f}s")
        print(f"  Words: {result['total_words']}, Rules: {len(rules)}")
        print(f"  Overall FPR: {result.get('fpr', 0.0):.4f}")
        print(f"  Gen-Coverage: {result.get('gen_coverage', 0.0):.2%} ({result.get('gen_covered', 0)}/{result['total_words']})")
        print(f"  Coverage: {result['coverage']:.2%}")

        if rules:
            print(f"  Sample rules:")
            for r in rules[:3]:
                print(f"    {r['Pattern']} ({r['Covered_Count']} words)")

    # 总体统计
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print("Overall Statistics")
    print(f"{'='*80}")
    print(f"Total Time: {elapsed:.2f}s ({elapsed/len(results):.2f}s per fragment)")
    print(f"Total Words: {total_stats['total_words']}")
    print(f"Covered Words: {total_stats['covered_words']}")
    print(f"Overall Coverage: {total_stats['covered_words']/total_stats['total_words']:.2%}")
    print(f"Gen-Covered Words: {total_stats.get('gen_covered', 0)}")
    print(f"Overall Gen-Coverage: {total_stats.get('gen_covered', 0)/total_stats['total_words']:.2%}")
    print(f"Total Rules: {total_stats['total_rules']}")

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()