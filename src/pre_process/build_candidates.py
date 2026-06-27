from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np


DEFAULT_MAX_POSTING = 300
DEFAULT_MIN_ANCHOR_SCORE = 1.0
DEFAULT_MIN_AA_LEN = 3
DEFAULT_MAX_LENGTH_RATIO = 5


def build_candidate_pairs(
    anchor2words: Dict[str, List[int]],
    id2word: List[str],
    max_posting: int = DEFAULT_MAX_POSTING,
    min_anchor_score: float = DEFAULT_MIN_ANCHOR_SCORE,
    min_aa_len: int = DEFAULT_MIN_AA_LEN,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """Build candidate pairs from anchor posting lists."""
    pair_scores: Dict[Tuple[int, int], float] = defaultdict(float)
    
    from .build_anchors import ANCHOR_WEIGHTS, aa_only
    
    for anchor_key, word_ids in anchor2words.items():
        if len(word_ids) > max_posting:
            continue
        
        anchor_type = anchor_key.split(':')[0]
        weight = ANCHOR_WEIGHTS.get(anchor_type, 0.5)
        
        for i in range(len(word_ids)):
            for j in range(i + 1, len(word_ids)):
                w1_id = word_ids[i]
                w2_id = word_ids[j]
                if w1_id != w2_id:
                    pair_scores[(w1_id, w2_id)] += weight
    
    candidate_pairs = []
    pair_scores_filtered = {}
    
    for (i, j), score in pair_scores.items():
        if score < min_anchor_score:
            continue
        
        w1 = id2word[i]
        w2 = id2word[j]
        
        aa1 = aa_only(w1)
        aa2 = aa_only(w2)
        
        len1 = len(aa1)
        len2 = len(aa2)
        
        if min(len1, len2) < min_aa_len:
            continue
        
        if max(len1, len2) / min(len1, len2) > max_length_ratio:
            continue
        
        candidate_pairs.append((i, j))
        pair_scores_filtered[(i, j)] = score
    
    return candidate_pairs, pair_scores_filtered


def batch_candidates(candidate_pairs: List[Tuple[int, int]], batch_size: int) -> List[List[Tuple[int, int]]]:
    """Split candidate pairs into batches."""
    batches = []
    for i in range(0, len(candidate_pairs), batch_size):
        batches.append(candidate_pairs[i:i+batch_size])
    return batches