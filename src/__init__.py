"""
引擎模块
"""

from .data_classes import MotifCandidate, OptimizationResult, FragmentResult, PipelineResult
from .motif_utils import (
    convert_motif_to_regex, match_words_with_motif,
    compute_wildcard_ratio, is_valid_wildcard_ratio,
    levenshtein_distance, align_to_longest,
    compute_position_entropy
)
from .strategies import (
    strategy_msa, strategy_frequency, strategy_clustering,
    strategy_greedy, strategy_chemical, strategy_aggressive
)
from .optimizer import (
    optimize_greedy, optimize_ilp, optimize_dp,
    compute_pareto_frontier, select_recommended
)
from .pipeline import process_fragment, run_pipeline
from .main import main

__all__ = [
    'MotifCandidate', 'OptimizationResult', 'FragmentResult', 'PipelineResult',
    'convert_motif_to_regex', 'match_words_with_motif',
    'compute_wildcard_ratio', 'is_valid_wildcard_ratio',
    'levenshtein_distance', 'align_to_longest', 'compute_position_entropy',
    'strategy_msa', 'strategy_frequency', 'strategy_clustering',
    'strategy_greedy', 'strategy_chemical', 'strategy_aggressive',
    'optimize_greedy', 'optimize_ilp', 'optimize_dp',
    'compute_pareto_frontier', 'select_recommended',
    'process_fragment', 'run_pipeline', 'main',
]
