"""
完整的处理pipeline
整合预处理、泛化、验证等模块
"""
from typing import List, Dict, Set, Optional, Tuple
import pickle
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

from core import GeneralizationRule
from preprocess import preprocess_words, get_negative_sample_pool
from generalize import generate_rules_for_fragment
from validate import RuleValidator, validate_rule_set, compute_coverage_metrics


@dataclass
class FragmentResult:
    """单个fragment的处理结果"""
    fragment_id: str
    rules: List[Dict]
    coverage_metrics: Dict
    validation_metrics: Dict
    processing_time: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class PipelineResult:
    """整个pipeline的处理结果"""
    total_fragments: int
    successful_fragments: int
    total_rules: int
    average_coverage: float
    average_fpr: float
    total_processing_time: float
    fragment_results: List[FragmentResult]
    
    def to_dict(self):
        return {
            'total_fragments': self.total_fragments,
            'successful_fragments': self.successful_fragments,
            'total_rules': self.total_rules,
            'average_coverage': self.average_coverage,
            'average_fpr': self.average_fpr,
            'total_processing_time': self.total_processing_time,
            'fragment_results': [r.to_dict() for r in self.fragment_results]
        }


def load_data(data_dir: str = "dataset") -> Tuple[Dict, Dict, Dict]:
    """
    加载所有必要的数据文件
    """
    # 加载fragment到蛋白词的映射
    with open(f"{data_dir}/origin.pkl", 'rb') as f:
        frag_to_words = pickle.load(f)
    
    # 加载蛋白词-片段对的score
    with open(f"{data_dir}/pwfm_hard_1024_1024_processed_ww2_wordfrag2score.pkl", 'rb') as f:
        wordfrag2score = pickle.load(f)
    
    # 构建蛋白词到fragment的映射（用于负样本）
    word_to_frags = {}
    for key, score in wordfrag2score.items():
        parts = key.split()
        if len(parts) == 2:
            word, frag = parts
            if word not in word_to_frags:
                word_to_frags[word] = set()
            word_to_frags[word].add(frag)
    
    return frag_to_words, wordfrag2score, word_to_frags


def process_single_fragment(fragment_id: str,
                           words: Set[str],
                           word_scores: Dict[str, float],
                           negative_pool: List[str],
                           max_time: float = 60.0,
                           target_coverage: float = 0.85,
                           max_fpr: float = 0.1) -> FragmentResult:
    """
    处理单个fragment
    """
    start_time = time.time()
    
    # 生成泛化规则
    rules = generate_rules_for_fragment(
        list(words),
        word_scores,
        max_time=max_time * 0.7  # 留出时间给验证
    )
    
    # 验证规则
    validator = RuleValidator(negative_pool)
    passed, validation_metrics = validator.validate_rule_set(rules, words)
    
    # 如果验证失败，尝试调整
    if not passed:
        # 降低标准，重新生成
        rules = generate_rules_for_fragment(
            list(words),
            word_scores,
            max_time=max_time * 0.5,
            max_rules=50
        )
        
        # 过滤规则
        rules = validator.filter_rules(rules)
        
        # 再次验证
        passed, validation_metrics = validator.validate_rule_set(rules, words)
    
    # 计算覆盖率指标
    coverage_metrics = compute_coverage_metrics(rules, words)
    
    # 转换规则为字典格式
    rules_dict = []
    for rule in rules:
        rule_dict = {
            'Pattern': str(rule),
            'Covered_Words': list(rule.covered_words),
            'Covered_Count': rule.covered_count,
            'Average_Score': rule.average_score,
            'False_Positive_Rate': validation_metrics.get('fpr', 0.0)
        }
        rules_dict.append(rule_dict)
    
    processing_time = time.time() - start_time
    
    return FragmentResult(
        fragment_id=fragment_id,
        rules=rules_dict,
        coverage_metrics=coverage_metrics,
        validation_metrics=validation_metrics,
        processing_time=processing_time
    )


def run_pipeline(data_dir: str = "dataset",
                output_dir: str = "results",
                max_time_per_fragment: float = 60.0,
                target_coverage: float = 0.85,
                max_fpr: float = 0.1,
                sample_size: int = 10000,
                max_fragments: Optional[int] = None) -> PipelineResult:
    """
    运行完整的pipeline
    """
    # 加载数据
    frag_to_words, wordfrag2score, word_to_frags = load_data(data_dir)
    
    # 限制处理的fragment数量（用于测试）
    if max_fragments:
        frag_to_words = dict(list(frag_to_words.items())[:max_fragments])
    
    # 准备负样本池
    all_words = set(word_to_frags.keys())
    negative_pool = []
    
    # 处理每个fragment
    fragment_results = []
    successful_count = 0
    total_rules = 0
    total_coverage = 0.0
    total_fpr = 0.0
    
    for i, (fragment_id, words) in enumerate(frag_to_words.items()):
        print(f"处理fragment {i+1}/{len(frag_to_words)}: {fragment_id}")
        
        # 获取该fragment对应的蛋白词score
        word_scores = {}
        for word in words:
            key = f"{word} {fragment_id}"
            word_scores[word] = wordfrag2score.get(key, 0.0)
        
        # 生成负样本池（排除当前fragment的词）
        current_negative_pool = get_negative_sample_pool(all_words, words, sample_size)
        
        # 处理单个fragment
        result = process_single_fragment(
            fragment_id,
            words,
            word_scores,
            current_negative_pool,
            max_time=max_time_per_fragment,
            target_coverage=target_coverage,
            max_fpr=max_fpr
        )
        
        fragment_results.append(result)
        
        # 统计
        if result.validation_metrics.get('coverage', 0) >= target_coverage:
            successful_count += 1
        
        total_rules += len(result.rules)
        total_coverage += result.coverage_metrics.get('coverage_ratio', 0)
        total_fpr += result.validation_metrics.get('fpr', 0)
    
    # 计算平均值
    n = len(fragment_results)
    avg_coverage = total_coverage / n if n > 0 else 0
    avg_fpr = total_fpr / n if n > 0 else 0
    
    # 创建最终结果
    pipeline_result = PipelineResult(
        total_fragments=len(frag_to_words),
        successful_fragments=successful_count,
        total_rules=total_rules,
        average_coverage=avg_coverage,
        average_fpr=avg_fpr,
        total_processing_time=sum(r.processing_time for r in fragment_results),
        fragment_results=fragment_results
    )
    
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "pipeline_results.json", 'w') as f:
        json.dump(pipeline_result.to_dict(), f, indent=2)
    
    # 保存详细规则
    detailed_rules = {}
    for result in fragment_results:
        detailed_rules[result.fragment_id] = result.rules
    
    with open(output_path / "detailed_rules.pkl", 'wb') as f:
        pickle.dump(detailed_rules, f)
    
    print(f"\n处理完成！")
    print(f"总fragments: {pipeline_result.total_fragments}")
    print(f"成功fragments: {pipeline_result.successful_fragments}")
    print(f"平均覆盖率: {pipeline_result.average_coverage:.2%}")
    print(f"平均FPR: {pipeline_result.average_fpr:.4f}")
    print(f"总规则数: {pipeline_result.total_rules}")
    print(f"总处理时间: {pipeline_result.total_processing_time:.1f}s")
    
    return pipeline_result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='蛋白词-分子片段规则优化')
    parser.add_argument('--data_dir', default='dataset', help='数据目录')
    parser.add_argument('--output_dir', default='results', help='输出目录')
    parser.add_argument('--max_time', type=float, default=60.0, help='单个fragment最大处理时间（秒）')
    parser.add_argument('--target_coverage', type=float, default=0.85, help='目标覆盖率')
    parser.add_argument('--max_fpr', type=float, default=0.1, help='最大假阳性率')
    parser.add_argument('--sample_size', type=int, default=10000, help='负样本池大小')
    parser.add_argument('--max_fragments', type=int, default=None, help='最多处理的fragment数量（用于测试）')
    
    args = parser.parse_args()
    
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_time_per_fragment=args.max_time,
        target_coverage=args.target_coverage,
        max_fpr=args.max_fpr,
        sample_size=args.sample_size,
        max_fragments=args.max_fragments
    )


if __name__ == '__main__':
    main()