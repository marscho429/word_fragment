import pickle
import os
from typing import Dict, List, Any


def load_frag2words(input_path: str) -> Dict[str, List[str]]:
    """Load frag2words from origin.pkl with field name compatibility."""
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        if 'frag2words' in data:
            return data['frag2words']
        if 'frag_to_words' in data:
            return data['frag_to_words']
        if 'fragment_to_words' in data:
            return data['fragment_to_words']
        if all(isinstance(k, str) and k.startswith('frag_') for k in data.keys()):
            return data
    
    raise ValueError(f"Unknown data format in {input_path}")


def build_word_index(frag2words: Dict[str, List[str]]) -> Dict[str, Any]:
    """Build word2id, id2word, word2frags, frag2word_ids."""
    word2id: Dict[str, int] = {}
    id2word: List[str] = []
    
    for words in frag2words.values():
        for w in words:
            if w not in word2id:
                word2id[w] = len(id2word)
                id2word.append(w)
    
    num_words = len(id2word)
    word2frags: Dict[int, List[str]] = {i: [] for i in range(num_words)}
    frag2word_ids: Dict[str, List[int]] = {}
    
    for frag_id, words in frag2words.items():
        word_ids = []
        for w in words:
            idx = word2id[w]
            word_ids.append(idx)
            if frag_id not in word2frags[idx]:
                word2frags[idx].append(frag_id)
        frag2word_ids[frag_id] = word_ids
    
    return {
        'word2id': word2id,
        'id2word': id2word,
        'word2frags': word2frags,
        'frag2word_ids': frag2word_ids
    }


def save_word_index(index: Dict[str, Any], output_path: str) -> None:
    """Save word_index to word_index.pkl."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_word_index(input_path: str) -> Dict[str, Any]:
    """Load word_index from word_index.pkl."""
    with open(input_path, 'rb') as f:
        return pickle.load(f)