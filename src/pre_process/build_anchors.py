from typing import Dict, List, Tuple

AA_ONLY_MAP = str.maketrans('', '', '_')

def aa_only(word: str) -> str:
    """Remove underscores from word."""
    return word.translate(AA_ONLY_MAP)

ANCHOR_WEIGHTS = {
    'exact3': 0.5,
    'exact4': 1.0,
    'exact5': 1.5,
    'prefix3': 0.3,
    'prefix4': 0.6,
    'prefix5': 1.0,
    'prefix6': 1.3,
    'suffix3': 0.3,
    'suffix4': 0.6,
    'suffix5': 1.0,
    'suffix6': 1.3,
    'spaced': 1.2,
    'gap_aware': 1.0,
}

def extract_exact_kmers(word: str) -> List[Tuple[str, float]]:
    """Extract exact k-mers for k=3,4,5 from aa_only(word)."""
    anchors = []
    aa = aa_only(word)
    for k in [3, 4, 5]:
        if len(aa) >= k:
            for i in range(len(aa) - k + 1):
                kmer = aa[i:i+k]
                key = f'exact{k}:{kmer}'
                anchors.append((key, ANCHOR_WEIGHTS[f'exact{k}']))
    return anchors

def extract_prefix_suffix(word: str) -> List[Tuple[str, float]]:
    """Extract prefix and suffix of lengths 3,4,5,6."""
    anchors = []
    aa = aa_only(word)
    for length in [3, 4, 5, 6]:
        if len(aa) >= length:
            prefix = aa[:length]
            key = f'prefix{length}:{prefix}'
            anchors.append((key, ANCHOR_WEIGHTS[f'prefix{length}']))
            
            suffix = aa[-length:]
            key = f'suffix{length}:{suffix}'
            anchors.append((key, ANCHOR_WEIGHTS[f'suffix{length}']))
    return anchors

def extract_spaced_anchors(word: str) -> List[Tuple[str, float]]:
    """Extract spaced anchors: left_len=3, right_len=3, gap=0..4."""
    anchors = []
    aa = aa_only(word)
    left_len = 3
    right_len = 3
    for gap in range(0, 5):
        total_len = left_len + gap + right_len
        if len(aa) >= total_len:
            for i in range(len(aa) - total_len + 1):
                left = aa[i:i+left_len]
                right = aa[i+left_len+gap:i+left_len+gap+right_len]
                pattern = f'{left}***{right}'
                key = f'spaced:{pattern}'
                anchors.append((key, ANCHOR_WEIGHTS['spaced']))
    return anchors

def extract_gap_aware_anchors(word: str) -> List[Tuple[str, float]]:
    """Extract gap-aware anchors for words containing underscores."""
    anchors = []
    if '_' not in word:
        return anchors
    
    parts = word.split('_')
    for i in range(len(parts) - 1):
        left = parts[i]
        right = parts[i+1]
        if len(left) >= 1 and len(right) >= 1:
            pattern = f'{left}***{right}'
            key = f'gap:{pattern}'
            anchors.append((key, ANCHOR_WEIGHTS['gap_aware']))
    
    return anchors

def extract_anchors(word: str) -> List[Tuple[str, float]]:
    """Extract all types of anchors for a word."""
    anchors = []
    anchors.extend(extract_exact_kmers(word))
    anchors.extend(extract_prefix_suffix(word))
    anchors.extend(extract_spaced_anchors(word))
    anchors.extend(extract_gap_aware_anchors(word))
    return anchors

def build_anchor2words(id2word: List[str]) -> Dict[str, List[int]]:
    """Build anchor2words mapping from all words."""
    anchor2words: Dict[str, List[int]] = {}
    for word_id, word in enumerate(id2word):
        anchors = extract_anchors(word)
        for key, _ in anchors:
            if key not in anchor2words:
                anchor2words[key] = []
            anchor2words[key].append(word_id)
    return anchor2words