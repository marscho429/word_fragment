/*
 * 规则生成器 - 凝聚式层次聚类 (Agglomerative Hierarchical Clustering)
 * 基于PDF方案：自底向上的规则感知凝聚式聚类
 *
 * 流程：
 * 1. 计算所有pair的DP alignment score，只保留score >= threshold的稀疏边
 * 2. 每个word保留topK高分边，形成稀疏pair graph
 * 3. 初始化cluster：每个word是一个singleton
 * 4. 用边初始化优先队列，执行规则感知凝聚式聚类
 * 5. 每个最终cluster生成一条consensus rule
 * 6. 每条rule反扫当前fragment的全部正样本，扩张covered words
 * 7. 对负样本计算FPR；贪心选择coverage高且FPR合格的规则
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <queue>
#include <set>
#include <chrono>
#include <cstring>
#include <cstdint>

namespace py = pybind11;

// ============================================================================
// Parameters
// ============================================================================
constexpr int GAP_MAX = 4;
constexpr int MAX_OPTIONAL_BLOCK = 4;
constexpr double PAIR_THRESHOLD = 2.0;
constexpr int TOPK_PER_WORD = 1000000;  // effectively no limit, keep all edges above threshold
constexpr int MAX_CLUSTER_SIZE = 50;
constexpr int MIN_CORE_POSITIONS = 3;
constexpr int MAX_SINGLE_WILDCARD = 4;
constexpr int MAX_WILDCARD_TOTAL = 6;
constexpr int MAX_TIER3_COUNT = 1;
constexpr int MAX_EXPLICIT_SET_SIZE = 3;
constexpr double MAX_COMPLEXITY = 12.0;
constexpr double TARGET_COVERAGE = 0.85;
constexpr double MAX_FPR = 0.1;

// ============================================================================
// Semantic Classes
// ============================================================================
struct SemanticClass {
    std::string name;
    std::string aas;
    int tier;
    double score;
};

const std::vector<SemanticClass> SEMANTIC_CLASSES = {
    {"acidic", "DE", 1, 0.9},
    {"basic", "KRH", 1, 0.9},
    {"sulfur", "CM", 1, 0.9},
    {"amide", "NQ", 1, 0.9},
    {"hydroxyl", "STY", 1, 0.9},
    {"tiny", "GAS", 1, 0.9},
    {"aromatic", "FYW", 2, 0.7},
    {"aliphatic", "AVLI", 2, 0.7},
    {"cyclic", "PFYWH", 2, 0.7},
    {"small", "GASCDPNT", 3, 0.5},
    {"nonpolar", "AVLIPFMWG", 3, 0.5},
    {"polar", "STNQYCDERKHR", 3, 0.5},
    {"bulky", "FYW RKEQMLI", 3, 0.5},
};

std::unordered_map<char, std::vector<const SemanticClass*>> aa_to_classes;

// ============================================================================
// Bitset
// ============================================================================
struct Bitset {
    std::vector<uint64_t> bits;
    int size;

    Bitset() : size(0) {}
    Bitset(int n) : size(n) {
        bits.resize((n + 63) / 64, 0);
    }

    void set(int i) {
        bits[i / 64] |= (1ULL << (i % 64));
    }

    bool test(int i) const {
        return (bits[i / 64] >> (i % 64)) & 1ULL;
    }

    void or_with(const Bitset& other) {
        for (size_t i = 0; i < bits.size(); ++i) {
            bits[i] |= other.bits[i];
        }
    }

    int count() const {
        int c = 0;
        for (auto b : bits) {
            c += __builtin_popcountll(b);
        }
        return c;
    }

    void reset() {
        std::fill(bits.begin(), bits.end(), 0);
    }
};

// ============================================================================
// Penalty Functions
// ============================================================================
double get_optional_penalty(int length) {
    if (length == 1) return -0.8;
    if (length == 2) return -1.2;
    if (length == 3) return -1.8;
    return -2.0 - (length - 4) * 0.5;
}

double get_flank_penalty(int flank_len) {
    if (flank_len == 0) return 0.0;
    if (flank_len <= 2) return -0.3;
    if (flank_len <= 4) return -0.5;
    return -1.0 - (flank_len - 4) * 0.3;
}

double aa_match_score(char qa, char ta) {
    if (qa == ta) return 1.0;
    if (qa == '_' || ta == '_') return 0.5;
    auto it = aa_to_classes.find(qa);
    if (it != aa_to_classes.end()) {
        for (const auto* sc : it->second) {
            if (sc->aas.find(ta) != std::string::npos) {
                return sc->score;
            }
        }
    }
    return 0.3;  // wildcard match score for mismatched chars
}

// ============================================================================
// DP Alignment (semi-global)
// ============================================================================
enum EdgeType { SAME_LENGTH, OPTIONAL_INDEL, CONTAINMENT, INTERNAL_GAP, UNDERSCORE };

struct AlignmentResult {
    double score;
    std::vector<std::string> tokens;  // rule tokens (without dashes)
    int edge_type;
};

AlignmentResult align_and_generate_rule(const std::string& query, const std::string& target,
                                         int max_optional_block, int gap_max) {
    const std::string& q_aa = query;
    const std::string& t_aa = target;
    int m = q_aa.size();
    int n = t_aa.size();

    // DP: dp[i][j] = best score for query[0..i-1] vs target[0..j-1]
    // trace: 0=match, 1=delete query, 2=insert target, 3=_ matches 1-4 target, 4=_ matches 1-4 query
    std::vector<std::vector<float>> dp(m + 1, std::vector<float>(n + 1, -1e9f));
    std::vector<std::vector<char>> trace(m + 1, std::vector<char>(n + 1, 0));
    std::vector<std::vector<int>> trace_len(m + 1, std::vector<int>(n + 1, 1));

    float del_penalty = (float)get_optional_penalty(1);

    dp[0][0] = 0.0f;
    for (int j = 1; j <= n; ++j) {
        dp[0][j] = (float)get_flank_penalty(j);
        trace[0][j] = 2;
    }
    for (int i = 1; i <= m; ++i) {
        dp[i][0] = dp[i-1][0] + del_penalty;
        trace[i][0] = 1;
    }

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            char qa = q_aa[i-1];
            char ta = t_aa[j-1];

            // Match (1 vs 1)
            float match_score = (float)aa_match_score(qa, ta);
            if (match_score > 0) {
                float score = dp[i-1][j-1] + match_score;
                if (score > dp[i][j]) {
                    dp[i][j] = score;
                    trace[i][j] = 0;
                    trace_len[i][j] = 1;
                }
            }

            // Delete query (optional)
            float del = dp[i-1][j] + del_penalty;
            if (del > dp[i][j]) {
                dp[i][j] = del;
                trace[i][j] = 1;
            }

            // Insert target (flank/gap)
            float ins = dp[i][j-1] + (float)get_flank_penalty(1);
            if (ins > dp[i][j]) {
                dp[i][j] = ins;
                trace[i][j] = 2;
            }

            // _ in query matches 1-4 target chars
            if (qa == '_' && ta != '_') {
                for (int k = 1; k <= 4 && k <= j; ++k) {
                    float score = dp[i-1][j-k] + 0.5f * k;
                    if (score > dp[i][j]) {
                        dp[i][j] = score;
                        trace[i][j] = 3;
                        trace_len[i][j] = k;
                    }
                }
            }

            // _ in target matches 1-4 query chars
            if (qa != '_' && ta == '_') {
                for (int k = 1; k <= 4 && k <= i; ++k) {
                    float score = dp[i-k][j-1] + 0.5f * k;
                    if (score > dp[i][j]) {
                        dp[i][j] = score;
                        trace[i][j] = 4;
                        trace_len[i][j] = k;
                    }
                }
            }
        }
    }

    // Find best end position (semi-global)
    float max_score = -1e9f;
    int best_j = n;
    for (int j = 0; j <= n; ++j) {
        float final_score = dp[m][j] + (float)get_flank_penalty(n - j);
        if (final_score > max_score) {
            max_score = final_score;
            best_j = j;
        }
    }

    // Traceback
    std::vector<std::string> rule_tokens;
    int i = m, j = best_j;
    int has_optional = 0, has_flank = 0, has_internal_gap = 0;
    bool seen_match = false;

    while (i > 0 || j > 0) {
        if (i == 0) {
            int flank_len = j;
            rule_tokens.push_back("x(0," + std::to_string(flank_len) + ")");
            has_flank = 1;
            j = 0;
            continue;
        }
        if (j == 0) {
            std::string optional;
            while (i > 0) {
                optional += q_aa[i-1];
                i--;
            }
            std::reverse(optional.begin(), optional.end());
            for (char c : optional) {
                rule_tokens.push_back("(" + std::string(1, c) + ")");
            }
            has_optional = 1;
            continue;
        }

        char t = trace[i][j];
        int tlen = trace_len[i][j];

        if (t == 0) {
            char qa = q_aa[i-1], ta = t_aa[j-1];
            seen_match = true;
            if (qa == '_' || ta == '_') {
                rule_tokens.push_back("_");
                i--; j--;
            } else if (qa == ta) {
                rule_tokens.push_back(std::string(1, qa));
                i--; j--;
            } else {
                bool found = false;
                for (const auto& sc : SEMANTIC_CLASSES) {
                    if (sc.aas.find(qa) != std::string::npos && sc.aas.find(ta) != std::string::npos) {
                        rule_tokens.push_back("[" + sc.name + "]");
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    rule_tokens.push_back("x(1,1)");
                }
                i--; j--;
            }
        } else if (t == 1) {
            std::string optional;
            int len = 0;
            while (i > 0 && trace[i][j] == 1) {
                optional += q_aa[i-1];
                len++;
                i--;
            }
            std::reverse(optional.begin(), optional.end());
            if (len <= max_optional_block) {
                for (char c : optional) {
                    rule_tokens.push_back("(" + std::string(1, c) + ")");
                }
                has_optional = 1;
            } else {
                rule_tokens.push_back("x(" + std::to_string(len) + "," + std::to_string(len) + ")");
                if (seen_match) has_internal_gap = 1;
                else has_flank = 1;
            }
        } else if (t == 2) {
            int len = 0;
            while (j > 0 && trace[i][j] == 2) {
                len++;
                j--;
            }
            if (len > 0) {
                rule_tokens.push_back("x(" + std::to_string(len) + "," + std::to_string(len) + ")");
                if (seen_match && i > 0) has_internal_gap = 1;
                else has_flank = 1;
            }
        } else if (t == 3) {
            rule_tokens.push_back("_");
            seen_match = true;
            i--; j -= tlen;
        } else if (t == 4) {
            rule_tokens.push_back("_");
            seen_match = true;
            i -= tlen; j--;
        }
    }

    std::reverse(rule_tokens.begin(), rule_tokens.end());

    // Add right flank: target characters after best_j, as range wildcard x(0,N)
    // so the rule matches both the query (no flank) and the target (with flank)
    if (best_j < n) {
        int flank_len = n - best_j;
        rule_tokens.push_back("x(0," + std::to_string(flank_len) + ")");
        has_flank = 1;
    }

    int edge_type = SAME_LENGTH;
    if (has_optional) edge_type = OPTIONAL_INDEL;
    else if (has_internal_gap) edge_type = INTERNAL_GAP;
    else if (has_flank) edge_type = CONTAINMENT;

    return {max_score, rule_tokens, edge_type};
}

// ============================================================================
// Rule Parsing and Matching
// ============================================================================
enum TokenType { LITERAL, SEMANTIC_CLASS, WILDCARD, FIXED_WILDCARD, RANGE_WILDCARD, OPTIONAL, OPTIONAL_CLASS };
struct Token {
    TokenType type;
    char aa;              // for LITERAL and OPTIONAL
    std::string class_name;  // for SEMANTIC_CLASS and OPTIONAL_CLASS
    int min_len;          // for wildcards
    int max_len;          // for wildcards
};

std::vector<Token> parse_rule_tokens(const std::string& rule) {
    std::vector<Token> tokens;
    std::string current;
    bool in_bracket = false;
    bool in_paren = false;

    auto push_literal = [&](const std::string& s) {
        if (s.size() == 1) {
            Token t;
            t.type = LITERAL;
            t.aa = s[0];
            t.min_len = 1; t.max_len = 1;
            tokens.push_back(t);
        } else {
            // multi-char literal (shouldn't happen normally)
            for (char c : s) {
                Token t;
                t.type = LITERAL;
                t.aa = c;
                t.min_len = 1; t.max_len = 1;
                tokens.push_back(t);
            }
        }
    };

    for (size_t i = 0; i < rule.size(); ++i) {
        char c = rule[i];

        if (c == '[') {
            if (!current.empty()) {
                push_literal(current);
                current.clear();
            }
            in_bracket = true;
            current = "";
        } else if (c == ']') {
            Token t;
            // Check if it's ([classname])
            if (in_paren) {
                // It's actually an optional class: ([classname])
                // The '(' was already consumed, so current is the classname
                t.type = OPTIONAL_CLASS;
                t.class_name = current;
                t.min_len = 1; t.max_len = 1;
            } else {
                t.type = SEMANTIC_CLASS;
                t.class_name = current;
                t.min_len = 1; t.max_len = 1;
            }
            tokens.push_back(t);
            current.clear();
            in_bracket = false;
        } else if (c == '(') {
            if (!current.empty() && !in_bracket && current != "x") {
                push_literal(current);
                current.clear();
            }
            in_paren = true;
            if (current == "x") {
                current += c;  // keep "x(" for x(...) syntax
            } else {
                current = "";
            }
        } else if (c == ')') {
            // Check if it's x(n,m) or x(n) or just (A) or ([class])
            if (current.size() >= 3 && current[0] == 'x' && current[1] == '(') {
                Token t;
                // Extract the part inside x(...)
                std::string inner = current.substr(2);  // skip "x("
                size_t comma = inner.find(',');
                if (comma != std::string::npos) {
                    t.type = RANGE_WILDCARD;
                    t.min_len = std::stoi(inner.substr(0, comma));
                    t.max_len = std::stoi(inner.substr(comma + 1));
                } else {
                    int len = std::stoi(inner);
                    if (len == 0) {
                        t.type = OPTIONAL;
                        t.aa = 0;
                        t.min_len = 0; t.max_len = 0;
                    } else {
                        t.type = FIXED_WILDCARD;
                        t.min_len = len;
                        t.max_len = len;
                    }
                }
                tokens.push_back(t);
            } else if (current.size() == 1) {
                Token t;
                t.type = OPTIONAL;
                t.aa = current[0];
                t.min_len = 1; t.max_len = 1;
                tokens.push_back(t);
            } else {
                // multi-char optional like (ABC) - treat as optional wildcard
                Token t;
                t.type = OPTIONAL;
                t.aa = current[0];  // first char
                t.min_len = 1; t.max_len = 1;
                tokens.push_back(t);
            }
            current.clear();
            in_paren = false;
        } else if (c == '_') {
            if (!current.empty() && !in_bracket) {
                push_literal(current);
                current.clear();
            }
            Token t;
            t.type = WILDCARD;
            t.min_len = 1; t.max_len = 4;
            tokens.push_back(t);
        } else if (c == '-') {
            if (!in_bracket && !in_paren) {
                if (!current.empty()) {
                    push_literal(current);
                    current.clear();
                }
            } else {
                current += '-';
            }
        } else {
            current += c;
        }
    }

    if (!current.empty()) {
        push_literal(current);
    }

    return tokens;
}

bool aa_in_semantic(const std::string& semantic, char aa) {
    for (const auto& sc : SEMANTIC_CLASSES) {
        if (sc.name == semantic) {
            return sc.aas.find(aa) != std::string::npos;
        }
    }
    return false;
}

bool rule_matches_word(const std::vector<Token>& tokens, const std::string& word) {
    int m = tokens.size();
    int n = word.size();

    if (m == 0) return n == 0;

    std::vector<std::vector<char>> dp(m + 1, std::vector<char>(n + 1, 0));
    dp[0][0] = 1;

    for (int i = 1; i <= m; ++i) {
        const Token& tok = tokens[i-1];
        for (int j = 0; j <= n; ++j) {
            if (!dp[i-1][j]) continue;

            if (tok.type == OPTIONAL) {
                // Match 0 chars
                dp[i][j] = 1;
                // Match 1 char
                if (j < n && word[j] == tok.aa) {
                    dp[i][j + 1] = 1;
                }
            } else if (tok.type == OPTIONAL_CLASS) {
                // Match 0 chars
                dp[i][j] = 1;
                // Match 1 char
                if (j < n && aa_in_semantic(tok.class_name, word[j])) {
                    dp[i][j + 1] = 1;
                }
            } else if (tok.type == LITERAL) {
                if (j < n && word[j] == tok.aa) {
                    dp[i][j + 1] = 1;
                }
            } else if (tok.type == SEMANTIC_CLASS) {
                if (j < n && aa_in_semantic(tok.class_name, word[j])) {
                    dp[i][j + 1] = 1;
                }
            } else if (tok.type == WILDCARD || tok.type == FIXED_WILDCARD || tok.type == RANGE_WILDCARD) {
                for (int len = tok.min_len; len <= tok.max_len && j + len <= n; ++len) {
                    dp[i][j + len] = 1;
                }
            }
        }
    }

    return dp[m][n];
}

bool rule_covers_word(const std::string& rule, const std::string& word) {
    auto tokens = parse_rule_tokens(rule);
    return rule_matches_word(tokens, word);
}

// ============================================================================
// Rule Complexity
// ============================================================================
double compute_rule_complexity(const std::vector<Token>& tokens) {
    double cx = 0.0;
    for (const auto& tok : tokens) {
        if (tok.type == LITERAL) cx += 1.0;
        else if (tok.type == SEMANTIC_CLASS) cx += 1.5;
        else if (tok.type == OPTIONAL) cx += 2.0;
        else if (tok.type == OPTIONAL_CLASS) cx += 2.5;
        else if (tok.type == WILDCARD) cx += 2.0;
        else if (tok.type == FIXED_WILDCARD) cx += 1.0 + tok.max_len * 0.5;
        else if (tok.type == RANGE_WILDCARD) cx += 1.0 + (tok.max_len - tok.min_len) * 0.5;
    }
    return cx;
}

int count_tier3(const std::vector<Token>& tokens) {
    int c = 0;
    for (const auto& tok : tokens) {
        if (tok.type == SEMANTIC_CLASS || tok.type == OPTIONAL_CLASS) {
            for (const auto& sc : SEMANTIC_CLASSES) {
                if (sc.name == tok.class_name && sc.tier == 3) { c++; break; }
            }
        }
    }
    return c;
}

int count_literal_core(const std::vector<Token>& tokens) {
    int c = 0;
    for (const auto& tok : tokens) {
        if (tok.type == LITERAL) c++;
    }
    return c;
}

int max_wildcard_len(const std::vector<Token>& tokens) {
    int mx = 0;
    for (const auto& tok : tokens) {
        if (tok.type == WILDCARD || tok.type == FIXED_WILDCARD || tok.type == RANGE_WILDCARD) {
            if (tok.max_len > mx) mx = tok.max_len;
        }
    }
    return mx;
}

int total_wildcard_len(const std::vector<Token>& tokens) {
    int total = 0;
    for (const auto& tok : tokens) {
        if (tok.type == WILDCARD || tok.type == FIXED_WILDCARD || tok.type == RANGE_WILDCARD) {
            total += tok.max_len;
        }
    }
    return total;
}

// ============================================================================
// Cluster
// ============================================================================
struct Cluster {
    int id;
    std::vector<int> word_indices;
    std::string rule;          // consensus rule
    std::vector<Token> tokens; // parsed consensus rule tokens
    double complexity;
    int center_word_idx;       // index of center word (best representative)
    bool active;

    Cluster() : id(-1), complexity(0), center_word_idx(-1), active(true) {}
    Cluster(int id_, int word_idx, const std::string& word)
        : id(id_), word_indices({word_idx}), rule(word), complexity(0), center_word_idx(word_idx), active(true) {}
};

// ============================================================================
// PairEdge
// ============================================================================
struct PairEdge {
    int i, j;
    double score;
    std::string rule;
    std::vector<Token> tokens;
    int edge_type;
    double complexity;

    PairEdge() : i(-1), j(-1), score(0), edge_type(0), complexity(0) {}
    PairEdge(int i_, int j_, double s, const std::string& r, const std::vector<Token>& tks, int et, double cx)
        : i(i_), j(j_), score(s), rule(r), tokens(tks), edge_type(et), complexity(cx) {}
};

bool operator<(const PairEdge& a, const PairEdge& b) {
    return a.score < b.score;  // max-heap
}

// ============================================================================
// Multi-Sequence Consensus Rule Generation
// ============================================================================

// Alignment columns for multi-sequence consensus
struct AlignmentColumns {
    // For each word (index 0 = center), for each center position, the word chars aligned to it
    std::vector<std::vector<std::vector<char>>> col_chars;
    // For each word, gaps between center positions: gaps[w][pos] = number of extra chars
    std::vector<std::vector<int>> gaps_before;  // gaps before each center position
    int center_len;
    std::vector<int> word_has_aa;  // for each word, which center positions have aligned AAs
};

// Align a word to the center word, return alignment mapping
// For each center position, returns which chars in the word align to it
AlignmentColumns align_words_to_center(
    const std::string& center,
    const std::vector<std::string>& words,
    const std::vector<int>& word_indices  // which words to align (all indices into cluster_words)
) {
    int n_words = word_indices.size();
    int center_len = center.size();
    
    AlignmentColumns cols;
    cols.center_len = center_len;
    cols.col_chars.resize(n_words, std::vector<std::vector<char>>(center_len));
    cols.gaps_before.resize(n_words, std::vector<int>(center_len + 1, 0));
    
    for (int wi = 0; wi < n_words; wi++) {
        const std::string& word = words[word_indices[wi]];
        
        // Run DP alignment between center (query) and word (target)
        int m = center.size();
        int n = word.size();
        
        std::vector<std::vector<double>> dp(m + 1, std::vector<double>(n + 1, -1e9));
        std::vector<std::vector<char>> trace(m + 1, std::vector<char>(n + 1, 0));
        std::vector<std::vector<int>> trace_len(m + 1, std::vector<int>(n + 1, 1));
        
        double del_penalty = get_optional_penalty(1);
        
        dp[0][0] = 0.0;
        for (int j = 1; j <= n; ++j) {
            dp[0][j] = get_flank_penalty(j);
            trace[0][j] = 2;
        }
        for (int i = 1; i <= m; ++i) {
            dp[i][0] = dp[i-1][0] + del_penalty;
            trace[i][0] = 1;
        }
        
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                char qa = center[i-1];
                char ta = word[j-1];
                
                double match_score = aa_match_score(qa, ta);
                if (match_score > 0) {
                    double score = dp[i-1][j-1] + match_score;
                    if (score > dp[i][j]) {
                        dp[i][j] = score;
                        trace[i][j] = 0;
                        trace_len[i][j] = 1;
                    }
                }
                
                double del = dp[i-1][j] + del_penalty;
                if (del > dp[i][j]) {
                    dp[i][j] = del;
                    trace[i][j] = 1;
                }
                
                double ins = dp[i][j-1] + get_flank_penalty(1);
                if (ins > dp[i][j]) {
                    dp[i][j] = ins;
                    trace[i][j] = 2;
                }
                
                if (qa == '_' && ta != '_') {
                    for (int k = 1; k <= 4 && k <= j; ++k) {
                        double score = dp[i-1][j-k] + 0.5 * k;
                        if (score > dp[i][j]) {
                            dp[i][j] = score;
                            trace[i][j] = 3;
                            trace_len[i][j] = k;
                        }
                    }
                }
                
                if (qa != '_' && ta == '_') {
                    for (int k = 1; k <= 4 && k <= i; ++k) {
                        double score = dp[i-k][j-1] + 0.5 * k;
                        if (score > dp[i][j]) {
                            dp[i][j] = score;
                            trace[i][j] = 4;
                            trace_len[i][j] = k;
                        }
                    }
                }
            }
        }
        
        // Find best end position
        double max_score = -1e9;
        int best_j = n;
        for (int j = 0; j <= n; ++j) {
            double final_score = dp[m][j] + get_flank_penalty(n - j);
            if (final_score > max_score) {
                max_score = final_score;
                best_j = j;
            }
        }
        
        // Traceback to extract alignment columns
        int i = m, j = best_j;
        int prev_center_pos = m;
        
        while (i > 0 || j > 0) {
            if (i == 0) {
                // Flank at end - remaining target chars
                cols.gaps_before[wi][0] += j;
                j = 0;
                continue;
            }
            if (j == 0) {
                // Deletion - center char not aligned to any word char
                i--;
                prev_center_pos = i;
                continue;
            }
            
            char t = trace[i][j];
            int tlen = trace_len[i][j];
            
            if (t == 0) {
                // Match: center[i-1] aligns with word[j-1]
                cols.col_chars[wi][i-1].push_back(word[j-1]);
                prev_center_pos = i - 1;
                i--; j--;
            } else if (t == 1) {
                // Deletion in query (center char is optional in word)
                i--;
                prev_center_pos = i;
            } else if (t == 2) {
                // Insertion in target (word char between center positions)
                if (prev_center_pos < m) {
                    cols.gaps_before[wi][prev_center_pos + 1]++;
                } else {
                    cols.gaps_before[wi][0]++;
                }
                j--;
            } else if (t == 3) {
                // _ in center matches tlen word chars
                for (int k = 0; k < tlen; k++) {
                    cols.col_chars[wi][i-1].push_back(word[j - tlen + k]);
                }
                prev_center_pos = i - 1;
                i--; j -= tlen;
            } else if (t == 4) {
                // _ in word matches tlen center chars
                for (int k = 0; k < tlen; k++) {
                    cols.col_chars[wi][i - tlen + k].push_back('_');
                }
                prev_center_pos = i - tlen;
                i -= tlen; j--;
            }
        }
    }
    
    return cols;
}

// Find the best semantic class covering a set of AAs
// Returns the class name, or empty if none found
// Prefers tier 1 over tier 2 over tier 3
std::string find_best_semantic_class(const std::set<char>& aas) {
    if (aas.empty()) return "";
    
    const SemanticClass* best = nullptr;
    for (const auto& sc : SEMANTIC_CLASSES) {
        bool covers_all = true;
        for (char aa : aas) {
            if (aa != '_' && sc.aas.find(aa) == std::string::npos) {
                covers_all = false;
                break;
            }
        }
        if (covers_all) {
            if (!best || sc.tier < best->tier ||
                (sc.tier == best->tier && sc.aas.size() < best->aas.size())) {
                best = &sc;
            }
        }
    }
    return best ? best->name : "";
}

// Generate consensus token from a column of AAs
// Returns the token string (e.g., "A", "[acidic]", "(A)", "x(1,2)")
std::string generate_column_token(
    const std::set<char>& aas,
    int num_words,
    int present_count
) {
    if (aas.empty()) {
        return "x(0,0)";
    }
    
    // If the set contains '_' (underscore), it's a wildcard placeholder
    // The '_' in the data represents 1-4 amino acids, so use _ as the token
    if (aas.count('_') > 0) {
        return "_";
    }
    
    // If all words have the same AA
    if (aas.size() == 1) {
        char aa = *aas.begin();
        if (aa == '_') return "_";
        if (present_count < num_words) {
            return "(" + std::string(1, aa) + ")";  // optional
        }
        return std::string(1, aa);
    }
    
    // Try semantic class
    std::string sem_class = find_best_semantic_class(aas);
    if (!sem_class.empty()) {
        if (present_count < num_words) {
            return "([" + sem_class + "])";  // optional semantic
        }
        return "[" + sem_class + "]";
    }
    
    // If some words don't have this position, make it optional
    if (present_count < num_words) {
        // Too many different AAs for optional - use wildcard
        if (aas.size() <= MAX_EXPLICIT_SET_SIZE) {
            std::string opt = "(";
            for (char aa : aas) {
                if (aa != '_') opt += aa;
            }
            opt += ")";
            return opt;
        }
        return "x(0,1)";
    }
    
    // If too many different AAs, use wildcard
    if (aas.size() <= MAX_EXPLICIT_SET_SIZE) {
        // Use explicit set like [ABC]
        std::string set = "[";
        for (char aa : aas) {
            if (aa != '_') set += aa;
        }
        set += "]";
        return set;
    }
    
    return "x(1,1)";
}

// Convert a Token to its string representation
std::string token_to_string(const Token& t) {
    switch (t.type) {
        case TokenType::LITERAL: return std::string(1, t.aa);
        case TokenType::WILDCARD: return "_";
        case TokenType::FIXED_WILDCARD: return "x(" + std::to_string(t.min_len) + ")";
        case TokenType::RANGE_WILDCARD: return "x(" + std::to_string(t.min_len) + "," + std::to_string(t.max_len) + ")";
        case TokenType::SEMANTIC_CLASS: return "[" + t.class_name + "]";
        case TokenType::OPTIONAL: return "(" + std::string(1, t.aa) + ")";
        case TokenType::OPTIONAL_CLASS: return "([" + t.class_name + "])";
        default: return "?";
    }
}

// Check if a character matches a token
bool char_matches_token(const Token& t, char c) {
    switch (t.type) {
        case TokenType::LITERAL: return c == t.aa;
        case TokenType::SEMANTIC_CLASS: {
            for (const auto& sc : SEMANTIC_CLASSES) {
                if (sc.name == t.class_name) {
                    return sc.aas.find(c) != std::string::npos;
                }
            }
            return false;
        }
        case TokenType::OPTIONAL: return c == t.aa;
        case TokenType::OPTIONAL_CLASS: {
            for (const auto& sc : SEMANTIC_CLASSES) {
                if (sc.name == t.class_name) {
                    return sc.aas.find(c) != std::string::npos;
                }
            }
            return false;
        }
        case TokenType::WILDCARD:
        case TokenType::FIXED_WILDCARD:
        case TokenType::RANGE_WILDCARD:
            return true;  // single char always matches wildcard
        default: return false;
    }
}

// Try to generalize a token to cover a new character
// Returns true if the token can be generalized, false if not
// Modifies the token in place
bool generalize_token(Token& t, char new_char) {
    // If the token already matches, no change needed
    if (char_matches_token(t, new_char)) return true;
    
    switch (t.type) {
        case TokenType::LITERAL: {
            // Try semantic class that covers both chars
            for (const auto& sc : SEMANTIC_CLASSES) {
                if (sc.aas.find(t.aa) != std::string::npos &&
                    sc.aas.find(new_char) != std::string::npos) {
                    t.type = TokenType::SEMANTIC_CLASS;
                    t.class_name = sc.name;
                    return true;
                }
            }
            // No class covers both -> use wildcard
            t.type = TokenType::WILDCARD;
            t.min_len = 1;
            t.max_len = 4;
            return true;
        }
        case TokenType::SEMANTIC_CLASS: {
            // Try a larger class that covers both
            for (const auto& sc : SEMANTIC_CLASSES) {
                // Check if the new class is a superset of the current class
                bool current_covered = true;
                const SemanticClass* current_sc = nullptr;
                for (const auto& sc2 : SEMANTIC_CLASSES) {
                    if (sc2.name == t.class_name) { current_sc = &sc2; break; }
                }
                if (!current_sc) break;
                
                for (char c : current_sc->aas) {
                    if (sc.aas.find(c) == std::string::npos) { current_covered = false; break; }
                }
                if (current_covered && sc.aas.find(new_char) != std::string::npos) {
                    t.class_name = sc.name;
                    return true;
                }
            }
            // No larger class -> use wildcard
            t.type = TokenType::WILDCARD;
            t.min_len = 1;
            t.max_len = 4;
            return true;
        }
        case TokenType::OPTIONAL: {
            // Try to make it required if it matches the new char
            if (new_char == t.aa) {
                t.type = TokenType::LITERAL;
                return true;
            }
            // Try semantic class
            for (const auto& sc : SEMANTIC_CLASSES) {
                if (sc.aas.find(t.aa) != std::string::npos &&
                    sc.aas.find(new_char) != std::string::npos) {
                    t.type = TokenType::SEMANTIC_CLASS;
                    t.class_name = sc.name;
                    return true;
                }
            }
            // Use wildcard
            t.type = TokenType::WILDCARD;
            t.min_len = 1;
            t.max_len = 4;
            return true;
        }
        case TokenType::OPTIONAL_CLASS: {
            // Make it required if new char is in the class
            for (const auto& sc : SEMANTIC_CLASSES) {
                if (sc.name == t.class_name) {
                    if (sc.aas.find(new_char) != std::string::npos) {
                        t.type = TokenType::SEMANTIC_CLASS;
                        return true;
                    }
                }
            }
            // Try larger class
            for (const auto& sc : SEMANTIC_CLASSES) {
                const SemanticClass* current_sc = nullptr;
                for (const auto& sc2 : SEMANTIC_CLASSES) {
                    if (sc2.name == t.class_name) { current_sc = &sc2; break; }
                }
                if (!current_sc) break;
                bool current_covered = true;
                for (char c : current_sc->aas) {
                    if (sc.aas.find(c) == std::string::npos) { current_covered = false; break; }
                }
                if (current_covered && sc.aas.find(new_char) != std::string::npos) {
                    t.type = TokenType::SEMANTIC_CLASS;
                    t.class_name = sc.name;
                    return true;
                }
            }
            // Use wildcard
            t.type = TokenType::WILDCARD;
            t.min_len = 1;
            t.max_len = 4;
            return true;
        }
        default: return false;
    }
}

// Try to generalize a rule to cover a new word
// Aligns the word to the rule tokens using DP, then relaxes tokens that don't match
bool generalize_rule_to_word(std::vector<Token>& tokens, const std::string& word) {
    int nt = tokens.size();
    int nw = word.size();
    
    if (nt == 0 || nw == 0) return false;
    
    // DP[i][j] = maximum matches for tokens[0..i-1] and word[0..j-1]
    const int INF = -1e9;
    std::vector<std::vector<int>> dp(nt + 1, std::vector<int>(nw + 1, INF));
    std::vector<std::vector<int>> trace(nt + 1, std::vector<int>(nw + 1, -1));
    
    dp[0][0] = 0;
    
    // Initialize first row (skip optional tokens)
    for (int i = 1; i <= nt; i++) {
        if (tokens[i-1].type == TokenType::OPTIONAL || 
            tokens[i-1].type == TokenType::OPTIONAL_CLASS) {
            dp[i][0] = dp[i-1][0];
            trace[i][0] = 1;  // skip optional
        } else {
            break;
        }
    }
    
    for (int i = 1; i <= nt; i++) {
        for (int j = 1; j <= nw; j++) {
            const Token& t = tokens[i-1];
            char c = word[j-1];
            
            // Match: token matches this character
            if (char_matches_token(t, c)) {
                if (dp[i-1][j-1] != INF) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                    trace[i][j] = 0;  // match
                }
            }
            
            // Wildcard can consume multiple chars
            if ((t.type == TokenType::WILDCARD ||
                 t.type == TokenType::FIXED_WILDCARD ||
                 t.type == TokenType::RANGE_WILDCARD) && dp[i][j-1] != INF) {
                dp[i][j] = dp[i][j-1];
                trace[i][j] = 2;  // wildcard consume
            }
            
            // Skip optional token
            if ((t.type == TokenType::OPTIONAL || 
                 t.type == TokenType::OPTIONAL_CLASS) && dp[i-1][j] != INF) {
                if (dp[i-1][j] >= dp[i][j]) {
                    dp[i][j] = dp[i-1][j];
                    trace[i][j] = 1;  // skip optional
                }
            }
        }
    }
    
    if (dp[nt][nw] == INF) return false;
    
    // Traceback to find alignment, then generalize mismatched tokens
    int i = nt, j = nw;
    while (i > 0 && j > 0) {
        int tr = trace[i][j];
        if (tr == 0) {
            // Match: check if token matches word char
            if (!char_matches_token(tokens[i-1], word[j-1])) {
                // Need to generalize this token
                if (!generalize_token(tokens[i-1], word[j-1])) {
                    return false;
                }
            }
            i--; j--;
        } else if (tr == 1) {
            // Skip optional
            i--;
        } else if (tr == 2) {
            // Wildcard consumes char
            j--;
        } else {
            break;
        }
    }
    
    // Verify the generalized rule now matches the word
    return rule_matches_word(tokens, word);
}
// Fast-path: validate a cached pair rule for a 2-word cluster
bool validate_pair_rule(
    const std::string& w1, const std::string& w2,
    const std::vector<Token>& tokens,
    bool& covers_both
) {
    covers_both = rule_matches_word(tokens, w1) && rule_matches_word(tokens, w2);
    return covers_both;
}

// Strategy: start with a pairwise rule, then generalize to cover more words
bool build_consensus_rule(
    const std::vector<std::string>& words,
    const std::vector<int>& indices,
    std::string& rule_out,
    std::vector<Token>& tokens_out,
    double& complexity_out,
    int& center_word_idx_out,
    int verbosity = 0
) {
    int n = indices.size();
    
    if (n == 1) {
        rule_out = words[indices[0]];
        tokens_out = parse_rule_tokens(rule_out);
        complexity_out = compute_rule_complexity(tokens_out);
        center_word_idx_out = indices[0];
        return true;
    }
    
    // Collect cluster words
    std::vector<std::string> cluster_words;
    for (int idx : indices) cluster_words.push_back(words[idx]);
    
    // Find center word: highest average pairwise score
    int center_local_idx = 0;
    double best_avg = -1e9;
    for (int i = 0; i < n; i++) {
        double sum = 0;
        int count = 0;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                auto al = align_and_generate_rule(cluster_words[i], cluster_words[j], MAX_OPTIONAL_BLOCK, GAP_MAX);
                sum += al.score;
                count++;
            }
        }
        double avg = (count > 0) ? sum / count : 0;
        if (avg > best_avg) {
            best_avg = avg;
            center_local_idx = i;
        }
    }
    center_word_idx_out = indices[center_local_idx];
    
    // Try pairwise rules: for each pair of words, check if the rule covers all words
    // Also try to generalize the rule to cover words it doesn't match
    struct PairCandidate {
        int i, j;
        double score;
        std::string rule;
        std::vector<Token> tokens;
    };
    std::vector<PairCandidate> candidates;
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            auto al = align_and_generate_rule(cluster_words[i], cluster_words[j], MAX_OPTIONAL_BLOCK, GAP_MAX);
            std::string rule_str;
            for (size_t k = 0; k < al.tokens.size(); ++k) {
                if (k > 0) rule_str += "-";
                rule_str += al.tokens[k];
            }
            PairCandidate pc;
            pc.i = i;
            pc.j = j;
            pc.score = al.score;
            pc.rule = rule_str;
            pc.tokens = parse_rule_tokens(rule_str);
            candidates.push_back(pc);
        }
    }
    
    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const PairCandidate& a, const PairCandidate& b) { return a.score > b.score; });
    
    // Try each candidate: first check if it covers all words, then try to generalize
    for (const auto& cand : candidates) {
        std::vector<Token> tokens = cand.tokens;
        
        bool covers_all = true;
        std::vector<int> unmatched;
        for (int wi = 0; wi < n; wi++) {
            if (!rule_matches_word(tokens, cluster_words[wi])) {
                covers_all = false;
                unmatched.push_back(wi);
            }
        }
        
        if (!covers_all) {
            // Try to generalize the rule to cover unmatched words
            bool generalized = false;
            for (int wi : unmatched) {
                if (generalize_rule_to_word(tokens, cluster_words[wi])) {
                    generalized = true;
                } else {
                    generalized = false;
                    break;
                }
            }
            if (!generalized) continue;
        }
        
        // Validate constraints
        double cx = compute_rule_complexity(tokens);
        if (n > MAX_CLUSTER_SIZE) continue;
        if (count_literal_core(tokens) < MIN_CORE_POSITIONS) continue;
        if (max_wildcard_len(tokens) > MAX_SINGLE_WILDCARD) continue;
        if (total_wildcard_len(tokens) > MAX_WILDCARD_TOTAL) continue;
        if (count_tier3(tokens) > MAX_TIER3_COUNT) continue;
        if (cx > MAX_COMPLEXITY) continue;
        
        // Build rule string from tokens
        std::string rule_str;
        for (size_t k = 0; k < tokens.size(); ++k) {
            if (k > 0) rule_str += "-";
            rule_str += token_to_string(tokens[k]);
        }
        
        rule_out = rule_str;
        tokens_out = tokens;
        complexity_out = cx;
        return true;
    }
    
    return false;
}

// ============================================================================
// AHC-based Rule Generation
// ============================================================================

py::dict generate_rules_ahc(
    const std::vector<std::string>& words,
    const std::vector<std::string>& negative_pool,
    double target_coverage,
    double max_fpr,
    int verbosity
) {
    int n_words = words.size();

    if (verbosity > 0) {
        printf("AHC: %d words, negative_pool=%d, target_cov=%.2f, max_fpr=%.3f\n",
               n_words, (int)negative_pool.size(), target_coverage, max_fpr);
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Phase 1: Compute all pairwise scores, keep sparse edges
    // ========================================================================
    if (verbosity > 0) printf("Phase 1: Computing pair scores...\n");

    std::vector<std::vector<PairEdge>> adj(n_words);  // adjacency list per word
    int total_pairs = 0;  // for verbose only

    for (int i = 0; i < n_words; ++i) {
        for (int j = i + 1; j < n_words; ++j) {
            total_pairs++;
            auto align = align_and_generate_rule(words[i], words[j], MAX_OPTIONAL_BLOCK, GAP_MAX);

            if (align.score >= PAIR_THRESHOLD) {
                std::string rule_str;
                for (size_t k = 0; k < align.tokens.size(); ++k) {
                    if (k > 0) rule_str += "-";
                    rule_str += align.tokens[k];
                }
                auto tokens = parse_rule_tokens(rule_str);
                double cx = compute_rule_complexity(tokens);

                PairEdge edge(i, j, align.score, rule_str, tokens, align.edge_type, cx);
                adj[i].push_back(edge);
                adj[j].push_back(edge);
            }
        }
    }

    auto phase1_end = std::chrono::high_resolution_clock::now();
    auto phase1_time = std::chrono::duration_cast<std::chrono::milliseconds>(phase1_end - total_start).count();

    if (verbosity > 0) {
        int total_edges = 0;
        for (int i = 0; i < n_words; ++i) total_edges += adj[i].size();
        printf("Phase 1 done: %d edges (%.1fs)\n", total_edges / 2, phase1_time / 1000.0);
    }

    // ========================================================================
    // Phase 2: Keep only topK edges per word
    // ========================================================================
    if (verbosity > 0) printf("Phase 2: Pruning edges (topK=%d)...\n", TOPK_PER_WORD);

    for (int i = 0; i < n_words; ++i) {
        if ((int)adj[i].size() > TOPK_PER_WORD) {
            std::sort(adj[i].begin(), adj[i].end(),
                      [](const PairEdge& a, const PairEdge& b) { return a.score > b.score; });
            adj[i].resize(TOPK_PER_WORD);
        }
    }

    int total_edges = 0;
    for (int i = 0; i < n_words; ++i) total_edges += adj[i].size();
    if (verbosity > 0) printf("Phase 2 done: %d directed edges\n", total_edges);

    // ========================================================================
    // Phase 3: Initialize clusters
    // ========================================================================
    if (verbosity > 0) printf("Phase 3: Initializing clusters...\n");

    std::vector<Cluster> clusters(n_words);
    for (int i = 0; i < n_words; ++i) {
        clusters[i] = Cluster(i, i, words[i]);
    }

    // ========================================================================
    // Phase 4: Pair-based clustering
    // Iterate over sorted edges, try to merge each unclustered pair
    // ========================================================================
    if (verbosity > 0) printf("Phase 4: Pair-based clustering...\n");

    // Collect and sort all unique edges by score descending
    std::vector<PairEdge> sorted_edges;
    std::unordered_set<std::string> edge_seen;
    for (int i = 0; i < n_words; ++i) {
        for (const auto& e : adj[i]) {
            int a = std::min(e.i, e.j);
            int b = std::max(e.i, e.j);
            std::string key = std::to_string(a) + "_" + std::to_string(b);
            if (edge_seen.find(key) == edge_seen.end()) {
                edge_seen.insert(key);
                sorted_edges.push_back(e);
            }
        }
    }
    std::sort(sorted_edges.begin(), sorted_edges.end(),
              [](const PairEdge& a, const PairEdge& b) { return a.score > b.score; });

    if (verbosity > 0) printf("  Sorted %d unique edges\n", (int)sorted_edges.size());

    std::vector<bool> clustered(n_words, false);
    std::vector<Cluster> result_clusters;
    int merge_count = 0;

    for (int ei = 0; ei < (int)sorted_edges.size(); ++ei) {
        const auto& e = sorted_edges[ei];
        if (clustered[e.i] || clustered[e.j]) continue;
        if (e.score < PAIR_THRESHOLD) break;

        // Use cached tokens from the edge for fast validation
        std::vector<Token> fast_tokens = e.tokens;
        bool covers_both = false;
        bool success = false;
        std::string consensus_rule;
        std::vector<Token> consensus_tokens;
        double consensus_cx;
        int center_word_idx = e.i;

        if (validate_pair_rule(words[e.i], words[e.j], fast_tokens, covers_both)) {
            // Tokens already cover both words - just validate constraints
            double cx = compute_rule_complexity(fast_tokens);
            bool constr_ok = (count_literal_core(fast_tokens) >= MIN_CORE_POSITIONS) &&
                (max_wildcard_len(fast_tokens) <= MAX_SINGLE_WILDCARD) &&
                (total_wildcard_len(fast_tokens) <= MAX_WILDCARD_TOTAL) &&
                (count_tier3(fast_tokens) <= MAX_TIER3_COUNT) &&
                (cx <= MAX_COMPLEXITY);

            if (constr_ok) {
                success = true;
                consensus_tokens = fast_tokens;
                consensus_cx = cx;
            }
        }

        if (!success) {
            // Try to generalize
            std::vector<Token> gen_tokens = fast_tokens;
            bool generalized = false;
            if (!covers_both) {
                if (!rule_matches_word(gen_tokens, words[e.i])) {
                    generalized = generalize_rule_to_word(gen_tokens, words[e.i]);
                } else if (!rule_matches_word(gen_tokens, words[e.j])) {
                    generalized = generalize_rule_to_word(gen_tokens, words[e.j]);
                }
            }
            if (generalized) {
                double cx = compute_rule_complexity(gen_tokens);
                bool constr_ok = (count_literal_core(gen_tokens) >= MIN_CORE_POSITIONS) &&
                    (max_wildcard_len(gen_tokens) <= MAX_SINGLE_WILDCARD) &&
                    (total_wildcard_len(gen_tokens) <= MAX_WILDCARD_TOTAL) &&
                    (count_tier3(gen_tokens) <= MAX_TIER3_COUNT) &&
                    (cx <= MAX_COMPLEXITY);
                if (constr_ok) {
                    success = true;
                    consensus_tokens = gen_tokens;
                    consensus_cx = cx;
                }
            }
        }

        if (!success) {
            // Fall back to full build_consensus_rule
            std::vector<int> pair = {e.i, e.j};
            success = build_consensus_rule(words, pair,
                                           consensus_rule, consensus_tokens, consensus_cx,
                                           center_word_idx, verbosity);
        }

        if (success) {
            if (consensus_rule.empty()) {
                for (size_t k = 0; k < consensus_tokens.size(); ++k) {
                    if (k > 0) consensus_rule += "-";
                    consensus_rule += token_to_string(consensus_tokens[k]);
                }
            }
            clustered[e.i] = true;
            clustered[e.j] = true;
            Cluster c;
            c.id = result_clusters.size();
            c.word_indices = {e.i, e.j};
            c.rule = consensus_rule;
            c.tokens = consensus_tokens;
            c.complexity = consensus_cx;
            c.center_word_idx = center_word_idx;
            c.active = true;
            result_clusters.push_back(c);
            merge_count++;
        }
    }

    // Remaining unclustered words become singletons
    for (int i = 0; i < n_words; ++i) {
        if (!clustered[i]) {
            Cluster c;
            c.id = result_clusters.size();
            c.word_indices = {i};
            c.rule = words[i];
            c.tokens = parse_rule_tokens(words[i]);
            c.complexity = compute_rule_complexity(c.tokens);
            c.center_word_idx = i;
            c.active = true;
            result_clusters.push_back(c);
        }
    }

    auto phase4_end = std::chrono::high_resolution_clock::now();
    auto phase4_time = std::chrono::duration_cast<std::chrono::milliseconds>(phase4_end - phase1_end).count();

    if (verbosity > 0) {
        int clustered_count = 0, singleton_count = 0;
        for (const auto& c : result_clusters) {
            if (c.word_indices.size() > 1) clustered_count += c.word_indices.size();
            else singleton_count++;
        }
        printf("Phase 4 done: %d clusters (%d merged words, %d singletons, %.1fs)\n",
               (int)result_clusters.size(), clustered_count, singleton_count, phase4_time / 1000.0);
    }

    // ========================================================================
    // Phase 5: Use result_clusters directly
    // ========================================================================
    std::vector<Cluster> final_clusters = result_clusters;

    if (verbosity > 0) printf("Phase 5: %d final clusters\n", (int)final_clusters.size());

    // ========================================================================
    // Phase 6: Scan rules against fragment positive samples
    // ========================================================================
    if (verbosity > 0) printf("Phase 6: Scanning rules against words...\n");

    struct RuleCandidate {
        std::string rule;
        std::vector<Token> tokens;
        double complexity;
        Bitset pos_bits;
        std::vector<std::string> covered_words;
    };

    std::vector<RuleCandidate> candidates;

    for (const auto& c : final_clusters) {
        RuleCandidate rc;
        rc.rule = c.rule;
        rc.tokens = c.tokens;
        rc.complexity = c.complexity;
        rc.pos_bits = Bitset(n_words);

        // Check which words this rule covers
        for (int i = 0; i < n_words; ++i) {
            if (rule_matches_word(rc.tokens, words[i])) {
                rc.pos_bits.set(i);
                rc.covered_words.push_back(words[i]);
            }
        }

        candidates.push_back(rc);
    }

    // ========================================================================
    // Phase 7: Compute FPR against negative samples
    // ========================================================================
    if (verbosity > 0) printf("Phase 7: Computing FPR...\n");

    int neg_size = negative_pool.size();
    Bitset neg_union(neg_size);

    for (auto& rc : candidates) {
        Bitset neg_bits(neg_size);
        for (int i = 0; i < neg_size; ++i) {
            if (rule_matches_word(rc.tokens, negative_pool[i])) {
                neg_bits.set(i);
            }
        }
        neg_union.or_with(neg_bits);
    }

    double overall_fpr = neg_size > 0 ? (double)neg_union.count() / neg_size : 0.0;

    if (verbosity > 0) {
        printf("  Overall FPR: %.4f (%d/%d)\n", overall_fpr, neg_union.count(), neg_size);
    }

    // ========================================================================
    // Phase 8: Greedy selection of final rules
    // ========================================================================
    if (verbosity > 0) printf("Phase 8: Greedy selection...\n");

    // Sort candidates by coverage (desc), then complexity (asc)
    std::sort(candidates.begin(), candidates.end(),
              [](const RuleCandidate& a, const RuleCandidate& b) {
                  int ca = a.pos_bits.count();
                  int cb = b.pos_bits.count();
                  if (ca != cb) return ca > cb;
                  return a.complexity < b.complexity;
              });

    Bitset covered(n_words);
    Bitset selected_neg(neg_size);
    std::vector<std::string> selected_patterns;
    std::vector<double> selected_scores;
    std::vector<std::vector<std::string>> covered_words_list;
    std::vector<int> covered_counts;
    int generalization_covered = 0;  // words covered by multi-word rules
    Bitset gen_covered(n_words);  // track which words are generalization-covered

    for (auto& rc : candidates) {
        // Count how many NEW words this rule covers (not already covered)
        int new_coverage = 0;
        for (int i = 0; i < n_words; ++i) {
            if (rc.pos_bits.test(i) && !covered.test(i)) new_coverage++;
        }
        if (new_coverage <= 0) continue;

        // Check FPR if we add this rule
        Bitset temp_neg = selected_neg;
        // Compute neg bits for this rule
        Bitset rc_neg(neg_size);
        for (int i = 0; i < neg_size; ++i) {
            if (rule_matches_word(rc.tokens, negative_pool[i])) {
                rc_neg.set(i);
            }
        }
        temp_neg.or_with(rc_neg);

        double new_fpr = neg_size > 0 ? (double)temp_neg.count() / neg_size : 0.0;
        if (new_fpr > max_fpr) {
            // Skip this rule, but try the next one
            continue;
        }

        // Accept this rule
        covered.or_with(rc.pos_bits);
        selected_neg.or_with(rc_neg);

        selected_patterns.push_back(rc.rule);
        selected_scores.push_back(rc.complexity);
        covered_counts.push_back(new_coverage);
        // Count total words this rule covers (for generalization coverage)
        int rule_total = rc.pos_bits.count();
        if (rule_total > 1) {
            // Mark all words covered by this rule as generalization-covered
            for (int i = 0; i < n_words; ++i) {
                if (rc.pos_bits.test(i) && !gen_covered.test(i)) {
                    gen_covered.set(i);
                }
            }
        }

        // Collect newly covered words
        std::vector<std::string> new_words;
        for (int i = 0; i < n_words; ++i) {
            if (rc.pos_bits.test(i)) {
                bool already = false;
                for (const auto& prev : covered_words_list) {
                    for (const auto& w : prev) {
                        if (w == words[i]) { already = true; break; }
                    }
                    if (already) break;
                }
                if (!already) new_words.push_back(words[i]);
            }
        }
        covered_words_list.push_back(new_words);

        if ((double)covered.count() / n_words >= target_coverage) break;
    }

    // ========================================================================
    // Phase 9: Add exact-match rules for uncovered words
    // ========================================================================
    for (int i = 0; i < n_words; ++i) {
        if (!covered.test(i)) {
            selected_patterns.push_back(words[i]);
            selected_scores.push_back(0.0);
            covered_words_list.push_back({words[i]});
            covered_counts.push_back(1);
            covered.set(i);
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    double coverage = (double)covered.count() / n_words;
    int gen_count = gen_covered.count();
    double gen_coverage = (double)gen_count / n_words;

    if (verbosity > 0) {
        printf("\n=== Results ===\n");
        printf("Overall FPR: %.4f (%d/%d)\n", overall_fpr, neg_union.count(), neg_size);
        printf("Coverage: %.2f%% (%d/%d)\n", coverage * 100, covered.count(), n_words);
        printf("Gen-Coverage: %.2f%% (%d/%d) [multi-word rules]\n", gen_coverage * 100, gen_count, n_words);
        printf("Rules: %d\n", (int)selected_patterns.size());
        printf("Total time: %.1fs\n", total_time / 1000.0);
    }

    // Build result
    py::dict result;
    result["patterns"] = selected_patterns;
    result["scores"] = selected_scores;
    result["covered_words_list"] = covered_words_list;
    result["covered_counts"] = covered_counts;
    result["total_covered"] = covered.count();
    result["total_words"] = n_words;
    result["coverage"] = coverage;
    result["gen_coverage"] = gen_coverage;
    result["gen_covered"] = gen_count;
    result["fpr"] = overall_fpr;

    return result;
}

// ============================================================================
// Python Bindings
// ============================================================================

PYBIND11_MODULE(rule_generator_cpp, m) {
    // Initialize semantic class lookup
    for (const auto& sc : SEMANTIC_CLASSES) {
        for (char aa : sc.aas) {
            if (aa != ' ') {
                aa_to_classes[aa].push_back(&sc);
            }
        }
    }

    m.def("generate_rules", &generate_rules_ahc,
          py::arg("words"),
          py::arg("negative_pool"),
          py::arg("target_coverage") = 0.85,
          py::arg("max_fpr") = 0.1,
          py::arg("verbosity") = 1,
          "Generate rules using agglomerative hierarchical clustering");

    m.def("rule_covers_word", &rule_covers_word,
          py::arg("rule"),
          py::arg("word"),
          "Check if a rule covers a word");

    m.def("align_pair", [](const std::string& a, const std::string& b) {
        auto result = align_and_generate_rule(a, b, MAX_OPTIONAL_BLOCK, GAP_MAX);
        py::dict res;
        res["score"] = result.score;
        std::string rule;
        for (size_t k = 0; k < result.tokens.size(); ++k) {
            if (k > 0) rule += "-";
            rule += result.tokens[k];
        }
        res["rule"] = rule;
        res["edge_type"] = result.edge_type;
        return res;
    }, "Align two words and generate rule");

    m.def("debug_aa_match_score", [](char qa, char ta) {
        double score = aa_match_score(qa, ta);
        py::dict res;
        res["qa"] = std::string(1, qa);
        res["ta"] = std::string(1, ta);
        res["score"] = score;
        py::list qa_classes;
        auto it = aa_to_classes.find(qa);
        if (it != aa_to_classes.end()) {
            for (const auto* sc : it->second) {
                qa_classes.append(sc->name);
            }
        }
        res["qa_classes"] = qa_classes;
        return res;
    }, "Debug aa_match_score");
}