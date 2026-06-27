#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>

namespace py = pybind11;

constexpr int GAP_MAX = 4;
constexpr int MAX_OPTIONAL_BLOCK = 4;

struct SemanticClass {
    std::string name;
    std::string aas;
    int tier;
    double score;
};

const std::vector<SemanticClass> SEMANTIC_CLASSES = {
    {"acidic", "DE", 1, 0.8},
    {"basic", "KRH", 1, 0.8},
    {"sulfur", "CM", 1, 0.8},
    {"amide", "NQ", 1, 0.8},
    {"hydroxyl", "STY", 1, 0.8},
    {"tiny", "GAS", 1, 0.8},
    {"aromatic", "FYW", 2, 0.6},
    {"aliphatic", "AVLI", 2, 0.6},
    {"cyclic", "PFYWH", 2, 0.55},
    {"small", "GASCDPNT", 3, 0.4},
    {"nonpolar", "AVLIPFMWG", 3, 0.35},
    {"polar", "STNQYCDERKHR", 3, 0.35},
    {"bulky", "FYW RKEQMLI", 3, 0.30},
};

std::unordered_map<char, std::vector<const SemanticClass*>> aa_to_classes;

void init_semantic_classes() {
    for (const auto& sc : SEMANTIC_CLASSES) {
        for (char aa : sc.aas) {
            if (aa != ' ') {
                aa_to_classes[aa].push_back(&sc);
            }
        }
    }
}

double get_semantic_score(char a, char b) {
    if (a == b) return 1.0;
    
    for (const auto& sc_a : aa_to_classes[a]) {
        for (const auto& sc_b : aa_to_classes[b]) {
            if (sc_a->name == sc_b->name) {
                return sc_a->score;
            }
        }
    }
    
    return -1.0;
}

double get_optional_penalty(int length) {
    if (length == 1) return -0.15;
    if (length == 2) return -0.40;
    if (length == 3) return -0.90;
    if (length == 4) return -1.60;
    return -2.0;
}

double get_flank_penalty(int length) {
    if (length <= 0) return 0.0;
    return -0.20 * length - 0.20;
}

double get_underscore_score(int block_len) {
    if (block_len >= 1 && block_len <= GAP_MAX) {
        return 0.10 * block_len;
    }
    return 0.0;
}

struct AlignmentResult {
    double score;
    std::string rule;
    int edge_type;
};

enum EdgeType {
    SAME_LENGTH_SEMANTIC = 0,
    OPTIONAL_INDEL = 1,
    CONTAINMENT = 2,
    INTERNAL_GAP = 3,
    UNDERSCORE_GAP = 4,
    MIXED_SEMIGLOBAL = 5
};

bool is_underscore(char c) {
    return c == '_';
}

bool is_amino_acid(char c) {
    return (c >= 'A' && c <= 'Z') && c != '_';
}

std::string aa_only(const std::string& word) {
    std::string result;
    for (char c : word) {
        if (is_amino_acid(c)) {
            result += c;
        }
    }
    return result;
}

struct DPState {
    double score;
    int prev_i;
    int prev_j;
    int type;
};

AlignmentResult score_embed(const std::string& query, const std::string& target, int max_optional_block, int gap_max);

AlignmentResult batch_score_and_rule(
    const std::vector<std::string>& words_a,
    const std::vector<std::string>& words_b,
    double score_threshold,
    int max_optional_block,
    int gap_max
) {
    std::vector<py::dict> results;
    
    for (size_t i = 0; i < words_a.size(); ++i) {
        const std::string& a = words_a[i];
        const std::string& b = words_b[i];
        
        AlignmentResult r1 = score_embed(a, b, max_optional_block, gap_max);
        AlignmentResult r2 = score_embed(b, a, max_optional_block, gap_max);
        
        AlignmentResult best;
        if (r1.score > r2.score) {
            best = r1;
        } else {
            best = r2;
        }
        
        if (best.score >= score_threshold) {
            py::dict d;
            d["index"] = (int)i;
            d["score"] = best.score;
            d["rule"] = best.rule;
            d["edge_type"] = best.edge_type;
            results.push_back(d);
        }
    }
    
    return results;
}

AlignmentResult score_embed(const std::string& query, const std::string& target, int max_optional_block, int gap_max) {
    std::string q_aa = aa_only(query);
    std::string t_aa = aa_only(target);
    
    int m = q_aa.size();
    int n = t_aa.size();
    
    if (m == 0 || n == 0) {
        return {0.0, "", MIXED_SEMIGLOBAL};
    }
    
    std::vector<std::vector<double>> dp(m + 1, std::vector<double>(n + 1, -std::numeric_limits<double>::max()));
    std::vector<std::vector<int>> trace(m + 1, std::vector<int>(n + 1, 0));
    
    for (int j = 0; j <= n; ++j) {
        dp[0][j] = get_flank_penalty(j);
        trace[0][j] = 1;
    }
    
    for (int i = 1; i <= m; ++i) {
        dp[i][0] = -std::numeric_limits<double>::max();
    }
    
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            double match = dp[i-1][j-1] + get_semantic_score(q_aa[i-1], t_aa[j-1]);
            
            double del = -std::numeric_limits<double>::max();
            for (int k = 1; k <= gap_max && i - k >= 0; ++k) {
                double penalty = get_optional_penalty(k);
                if (dp[i-k][j] + penalty > del) {
                    del = dp[i-k][j] + penalty;
                }
            }
            
            double ins = -std::numeric_limits<double>::max();
            for (int k = 1; k <= gap_max && j - k >= 0; ++k) {
                double penalty = get_optional_penalty(k);
                if (dp[i][j-k] + penalty > ins) {
                    ins = dp[i][j-k] + penalty;
                }
            }
            
            dp[i][j] = std::max({match, del, ins});
            
            if (dp[i][j] == match) trace[i][j] = 0;
            else if (dp[i][j] == del) trace[i][j] = 1;
            else trace[i][j] = 2;
        }
    }
    
    double max_score = -std::numeric_limits<double>::max();
    int best_j = n;
    
    for (int j = 0; j <= n; ++j) {
        double final_score = dp[m][j] + get_flank_penalty(n - j);
        if (final_score > max_score) {
            max_score = final_score;
            best_j = j;
        }
    }
    
    std::vector<std::string> rule_tokens;
    int i = m;
    int j = best_j;
    int has_optional = 0;
    int has_flank = 0;
    int has_internal_gap = 0;
    
    while (i > 0 || j > 0) {
        if (i == 0) {
            j--;
            has_flank = 1;
            continue;
        }
        if (j == 0) {
            i--;
            continue;
        }
        
        if (trace[i][j] == 0) {
            char qa = q_aa[i-1];
            char ta = t_aa[j-1];
            
            if (qa == ta) {
                rule_tokens.push_back(std::string(1, qa));
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
                    rule_tokens.push_back(std::string(1, qa));
                }
            }
            i--;
            j--;
        } else if (trace[i][j] == 1) {
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
                has_internal_gap = 1;
            }
        } else {
            j--;
            has_flank = 1;
        }
    }
    
    std::reverse(rule_tokens.begin(), rule_tokens.end());
    
    std::string rule;
    for (size_t k = 0; k < rule_tokens.size(); ++k) {
        if (k > 0) rule += "-";
        rule += rule_tokens[k];
    }
    
    int edge_type = SAME_LENGTH_SEMANTIC;
    if (has_optional) edge_type = OPTIONAL_INDEL;
    else if (has_flank) edge_type = CONTAINMENT;
    else if (has_internal_gap) edge_type = INTERNAL_GAP;
    
    return {max_score, rule, edge_type};
}

PYBIND11_MODULE(protein_score_cpp, m) {
    init_semantic_classes();
    
    m.def("batch_score_and_rule", &batch_score_and_rule,
          py::arg("words_a"),
          py::arg("words_b"),
          py::arg("score_threshold"),
          py::arg("max_optional_block") = 4,
          py::arg("gap_max") = 4,
          "Batch score and generate rules for word pairs");
}