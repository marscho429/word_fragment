"""
蛋白词-分子片段规则优化模块
"""

from .core import (
    RuleElement, GeneralizationRule, RuleElementType,
    TIERED_SEMANTIC_GROUPS, AMINO_ACID_TO_PROPERTIES, ALL_AMINO_ACIDS,
    get_best_property_for_amino_acids, parse_word,
    create_exact_element, create_property_element, create_optional_element,
    create_wildcard_fixed_element, create_wildcard_range_element,
    create_wildcard_any_element, create_choice_element
)

from .preprocess import (
    ProcessedWord, preprocess_word, preprocess_words,
    group_words_by_length, find_similar_words,
    calculate_amino_overlap, longest_common_subsequence_length,
    extract_common_pattern, cluster_words_by_pattern,
    find_word_pairs_with_high_overlap, get_negative_sample_pool
)

from .generalize import (
    AlignmentResult, align_words, generate_rule_from_alignment,
    mine_high_quality_rules, find_word_group,
    optimize_rule_coverage, generate_rules_for_fragment
)

from .validate import (
    RuleValidator, match_word_with_rule, validate_rule_against_original_words,
    calculate_fpr_for_rule, calculate_fpr_for_rule_set,
    evaluate_rule_quality, validate_rule_set, filter_rules_by_quality,
    compute_coverage_metrics
)

from .pipeline import (
    FragmentResult, PipelineResult, load_data,
    process_single_fragment, run_pipeline, main
)

__all__ = [
    # Core
    'RuleElement', 'GeneralizationRule', 'RuleElementType',
    'TIERED_SEMANTIC_GROUPS', 'AMINO_ACID_TO_PROPERTIES', 'ALL_AMINO_ACIDS',
    'get_best_property_for_amino_acids', 'parse_word',
    'create_exact_element', 'create_property_element', 'create_optional_element',
    'create_wildcard_fixed_element', 'create_wildcard_range_element',
    'create_wildcard_any_element', 'create_choice_element',
    
    # Preprocess
    'ProcessedWord', 'preprocess_word', 'preprocess_words',
    'group_words_by_length', 'find_similar_words',
    'calculate_amino_overlap', 'longest_common_subsequence_length',
    'extract_common_pattern', 'cluster_words_by_pattern',
    'find_word_pairs_with_high_overlap', 'get_negative_sample_pool',
    
    # Generalize
    'AlignmentResult', 'align_words', 'generate_rule_from_alignment',
    'mine_high_quality_rules', 'find_word_group',
    'optimize_rule_coverage', 'generate_rules_for_fragment',
    
    # Validate
    'RuleValidator', 'match_word_with_rule', 'validate_rule_against_original_words',
    'calculate_fpr_for_rule', 'calculate_fpr_for_rule_set',
    'evaluate_rule_quality', 'validate_rule_set', 'filter_rules_by_quality',
    'compute_coverage_metrics',
    
    # Pipeline
    'FragmentResult', 'PipelineResult', 'load_data',
    'process_single_fragment', 'run_pipeline', 'main',
]
