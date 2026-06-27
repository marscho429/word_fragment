import argparse
import os
import pickle
import numpy as np


EDGE_TYPES = {
    0: "same_length_semantic",
    1: "optional_indel",
    2: "containment",
    3: "internal_gap",
    4: "underscore_gap",
    5: "mixed_semiglobal"
}


def load_preprocess(preprocess_dir):
    word_index_path = os.path.join(preprocess_dir, 'word_index.pkl')
    neighbors_path = os.path.join(preprocess_dir, 'neighbors_topk.npz')
    rules_path = os.path.join(preprocess_dir, 'rules.pkl')
    
    with open(word_index_path, 'rb') as f:
        word_index = pickle.load(f)
    
    neighbors = np.load(neighbors_path)
    
    with open(rules_path, 'rb') as f:
        rules = pickle.load(f)
    
    return word_index, neighbors, rules


def inspect_word(word, preprocess_dir, top=20):
    word_index, neighbors, rules = load_preprocess(preprocess_dir)
    
    word2id = word_index['word2id']
    id2word = word_index['id2word']
    id2rule = rules['id2rule']
    
    indptr = neighbors['indptr']
    neighbor_id = neighbors['neighbor_id']
    score = neighbors['score']
    rule_id = neighbors['rule_id']
    edge_type = neighbors['edge_type']
    
    if word not in word2id:
        print(f"Word not found: {word}")
        return
    
    word_id = word2id[word]
    
    print(f"word: {word}")
    print(f"id: {word_id}")
    print()
    
    start = indptr[word_id]
    end = indptr[word_id + 1]
    
    print(f"neighbor score edge_type rule")
    count = 0
    for i in range(start, end):
        if count >= top:
            break
        n_id = neighbor_id[i]
        n_score = score[i]
        n_rule_id = rule_id[i]
        n_edge_type = edge_type[i]
        
        n_word = id2word[n_id]
        n_rule = id2rule[n_rule_id]
        n_edge_type_name = EDGE_TYPES.get(n_edge_type, f"unknown({n_edge_type})")
        
        print(f"{n_word} {n_score:.2f} {n_edge_type_name} {n_rule}")
        count += 1


def inspect_fragment(fragment_id, preprocess_dir):
    word_index, neighbors, rules = load_preprocess(preprocess_dir)
    
    frag2word_ids = word_index['frag2word_ids']
    
    if fragment_id not in frag2word_ids:
        print(f"Fragment not found: {fragment_id}")
        return
    
    word_ids = frag2word_ids[fragment_id]
    num_words = len(word_ids)
    
    indptr = neighbors['indptr']
    
    has_neighbor = 0
    num_induced_edges = 0
    
    for w_id in word_ids:
        start = indptr[w_id]
        end = indptr[w_id + 1]
        n_neighbors = end - start
        if n_neighbors > 0:
            has_neighbor += 1
            num_induced_edges += n_neighbors
    
    graph_coverable_upper_bound = has_neighbor / num_words if num_words > 0 else 0.0
    
    print(f"fragment: {fragment_id}")
    print(f"num_words: {num_words}")
    print(f"num_induced_edges: {num_induced_edges}")
    print(f"graph_coverable_upper_bound: {graph_coverable_upper_bound:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Inspect preprocess results')
    parser.add_argument('--preprocess', required=True, help='Preprocess directory')
    parser.add_argument('--word', help='Inspect specific word')
    parser.add_argument('--fragment', help='Inspect specific fragment')
    parser.add_argument('--top', type=int, default=20, help='Top neighbors to show')
    
    args = parser.parse_args()
    
    if args.word:
        inspect_word(args.word, args.preprocess, args.top)
    elif args.fragment:
        inspect_fragment(args.fragment, args.preprocess)
    else:
        print("Please specify either --word or --fragment")


if __name__ == '__main__':
    main()