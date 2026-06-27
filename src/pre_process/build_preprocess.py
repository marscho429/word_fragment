import argparse
import os
import sys
import json
import pickle
import time
import glob
from typing import Dict, List, Tuple, Any

import numpy as np
from tqdm import tqdm


EDGE_TYPES = {
    0: "same_length_semantic",
    1: "optional_indel",
    2: "containment",
    3: "internal_gap",
    4: "underscore_gap",
    5: "mixed_semiglobal"
}


def parse_args():
    parser = argparse.ArgumentParser(description='Build global pairwise preprocess for protein words')
    parser.add_argument('--input', required=True, help='Input origin.pkl file')
    parser.add_argument('--output', required=True, help='Output directory for preprocess files')
    parser.add_argument('--topk', type=int, default=100, help='TopK neighbors per word')
    parser.add_argument('--score-threshold', type=float, default=3.0, help='Minimum score threshold')
    parser.add_argument('--max-posting', type=int, default=300, help='Max posting list size for anchor')
    parser.add_argument('--batch-size', type=int, default=100000, help='Batch size for C++ scoring')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers (not used yet)')
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    
    os.makedirs(args.output, exist_ok=True)
    tmp_edges_dir = os.path.join(args.output, 'tmp_edges')
    os.makedirs(tmp_edges_dir, exist_ok=True)
    
    word_index_path = os.path.join(args.output, 'word_index.pkl')
    neighbors_path = os.path.join(args.output, 'neighbors_topk.npz')
    rules_path = os.path.join(args.output, 'rules.pkl')
    summary_path = os.path.join(args.output, 'preprocess_summary.json')
    
    from .io_utils import load_frag2words, build_word_index, save_word_index
    
    print("[1/5] Loading and indexing words...")
    frag2words = load_frag2words(args.input)
    index = build_word_index(frag2words)
    word2id = index['word2id']
    id2word = index['id2word']
    word2frags = index['word2frags']
    frag2word_ids = index['frag2word_ids']
    
    num_words = len(id2word)
    num_fragments = len(frag2word_ids)
    
    print(f"  num_fragments: {num_fragments}")
    print(f"  num_unique_words: {num_words}")
    
    save_word_index(index, word_index_path)
    print(f"  Saved word_index.pkl")
    
    print("[2/5] Extracting anchors...")
    from .build_anchors import build_anchor2words
    anchor2words = build_anchor2words(id2word)
    print(f"  num_anchors: {len(anchor2words)}")
    
    print("[3/5] Building candidate pairs...")
    from .build_candidates import build_candidate_pairs
    candidate_pairs, pair_scores = build_candidate_pairs(
        anchor2words, id2word,
        max_posting=args.max_posting
    )
    
    num_candidate_pairs = len(candidate_pairs)
    print(f"  num_candidate_pairs: {num_candidate_pairs}")
    
    batches = []
    for i in range(0, num_candidate_pairs, args.batch_size):
        batches.append(candidate_pairs[i:i+args.batch_size])
    print(f"  num_batches: {len(batches)}")
    
    print("[4/5] C++ scoring candidate pairs...")
    try:
        from . import protein_score_cpp
    except ImportError:
        print("ERROR: protein_score_cpp module not found.")
        print("Please compile first: bash src/pre_process/compile_cpp.sh")
        sys.exit(1)
    
    processed_pairs = 0
    accepted_edges = 0
    edge_type_counts = {name: 0 for name in EDGE_TYPES.values()}
    
    existing_batches = set()
    for f in glob.glob(os.path.join(tmp_edges_dir, 'edges_batch_*.pkl')):
        batch_num = int(os.path.basename(f).split('_')[2].split('.')[0])
        existing_batches.add(batch_num)
    
    print(f"  Found {len(existing_batches)} existing batch files (will skip)")
    
    for batch_idx, batch in tqdm(enumerate(batches), total=len(batches), desc="Scoring pairs"):
        if batch_idx in existing_batches:
            processed_pairs += len(batch)
            continue
        
        words_a = [id2word[i] for i, j in batch]
        words_b = [id2word[j] for i, j in batch]
        
        results = protein_score_cpp.batch_score_and_rule(
            words_a, words_b,
            score_threshold=args.score_threshold,
            max_optional_block=4,
            gap_max=4
        )
        
        batch_edges = []
        for r in results:
            idx = r['index']
            score = r['score']
            rule = r['rule']
            edge_type = r['edge_type']
            
            i, j = batch[idx]
            batch_edges.append((i, j, score, rule, edge_type))
            batch_edges.append((j, i, score, rule, edge_type))
            
            accepted_edges += 1
            edge_type_name = EDGE_TYPES.get(edge_type, "unknown")
            edge_type_counts[edge_type_name] += 1
        
        batch_filename = os.path.join(tmp_edges_dir, f'edges_batch_{batch_idx:06d}.pkl')
        with open(batch_filename, 'wb') as f:
            pickle.dump(batch_edges, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        processed_pairs += len(batch)
        
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            accept_rate = accepted_edges / processed_pairs if processed_pairs > 0 else 0
            print(f"  [Batch {batch_idx+1}/{len(batches)}] processed={processed_pairs}, accepted={accepted_edges}, "
                  f"rate={accept_rate:.3f}, elapsed={elapsed:.1f}s")
    
    print("[5/5] Building topK adjacency and saving...")
    
    print("  Loading all edge batches...")
    all_edges = []
    for f in tqdm(glob.glob(os.path.join(tmp_edges_dir, 'edges_batch_*.pkl')), desc="Loading edges"):
        with open(f, 'rb') as fp:
            edges = pickle.load(fp)
            all_edges.extend(edges)
    
    num_verified_edges = len(all_edges)
    print(f"  Total verified edges: {num_verified_edges}")
    
    print("  Building rule pool...")
    rule2id: Dict[str, int] = {}
    id2rule: List[str] = []
    
    for i, j, score, rule, edge_type in all_edges:
        if rule not in rule2id:
            rule2id[rule] = len(id2rule)
            id2rule.append(rule)
    
    print(f"  num_unique_rules: {len(id2rule)}")
    
    with open(rules_path, 'wb') as f:
        pickle.dump({'id2rule': id2rule, 'rule2id': rule2id}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved rules.pkl")
    
    print("  Building topK adjacency...")
    adjacency: Dict[int, List[Tuple[int, float, int, int]]] = {i: [] for i in range(num_words)}
    
    for i, j, score, rule, edge_type in all_edges:
        rule_id = rule2id[rule]
        adjacency[i].append((j, score, rule_id, edge_type))
    
    for i in range(num_words):
        adjacency[i].sort(key=lambda x: -x[1])
        adjacency[i] = adjacency[i][:args.topk]
    
    print("  Building CSR arrays...")
    indptr = np.zeros(num_words + 1, dtype=np.int64)
    for i in range(num_words):
        indptr[i+1] = indptr[i] + len(adjacency[i])
    
    num_edges_total = int(indptr[-1])
    neighbor_id = np.zeros(num_edges_total, dtype=np.int32)
    score = np.zeros(num_edges_total, dtype=np.float32)
    rule_id = np.zeros(num_edges_total, dtype=np.int32)
    edge_type = np.zeros(num_edges_total, dtype=np.uint8)
    
    pos = 0
    for i in range(num_words):
        for j, s, r_id, e_type in adjacency[i]:
            neighbor_id[pos] = j
            score[pos] = s
            rule_id[pos] = r_id
            edge_type[pos] = e_type
            pos += 1
    
    np.savez(neighbors_path,
             indptr=indptr,
             neighbor_id=neighbor_id,
             score=score,
             rule_id=rule_id,
             edge_type=edge_type)
    print(f"  Saved neighbors_topk.npz")
    
    print("  Writing summary...")
    runtime_seconds = time.time() - start_time
    summary = {
        "num_fragments": num_fragments,
        "num_unique_words": num_words,
        "num_candidate_pairs": num_candidate_pairs,
        "num_verified_edges": num_verified_edges,
        "topK": args.topk,
        "score_threshold": args.score_threshold,
        "runtime_seconds": runtime_seconds,
        "edge_type_counts": edge_type_counts
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved preprocess_summary.json")
    
    print(f"\nPreprocessing complete!")
    print(f"  Runtime: {runtime_seconds:.1f}s")
    print(f"  Coverage: {sum(1 for i in range(num_words) if len(adjacency[i]) > 0) / num_words * 100:.2f}%")


if __name__ == '__main__':
    main()