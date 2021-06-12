from random import choice, shuffle
from math import sqrt
import pandas as pd
import numpy as np

from memory_bound_pursuit import pursuit, online_pursuit_mc
from memory_bound_xsit import online_xsit, make_lexicon

ALL = ['a1', 'a2', 'b1', 'b2', 'c1', 'c2',
       'd1', 'd2', 'e1', 'e2', 'f1', 'f2',
       'g', 'h', 'i', 'j', 'k', 'l']
GOLD = {'A': ['a1', 'a2'],
        'B': ['b1', 'b2'],
        'C': ['c1', 'c2'],
        'D': ['d1', 'd2'],
        'E': ['e1', 'e2'],
        'F': ['f1', 'f2'],
        'G': ['g'],
        'H': ['h'],
        'I': ['i'],
        'J': ['j'],
        'K': ['k'],
        'L': ['l'],
        }
DOUBLE = {'A', 'B', 'C', 'D', 'E', 'F'}
DATA1 = "./data/yurovsky_yu_2008/hom_exp1.txt"
DATA2 = "./data/yurovsky_yu_2008/hom_exp2.txt"


def mc(meanings):
    def mc_function(lexicon, word, options, associations):
        return online_pursuit_mc(word, options, lexicon, meanings, associations)

    return mc_function


def run_one_word(lexicon, associations, function, word, initial_options):
    options = initial_options.copy()
    others = ALL.copy()
    for meaning in GOLD[word]:
        others.remove(meaning)
    other_options = 4 - len(initial_options)
    for i in range(other_options):
        new_word = choice(others)
        options.append(new_word)
        others.remove(new_word)
    shuffle(options)
    answer = function(lexicon, word, options, associations)
    correct = 0
    prim_rec = None
    if answer in initial_options:
        correct = 1
        prim_rec = initial_options.index(answer)
    return correct, prim_rec


def run_one_trial_primacy(associations, function, lexicon):
    double_count = 0
    single_count = 0
    for word in GOLD:
        options = [GOLD[word][0]]
        answer = run_one_word(lexicon, associations, function, word, options)[0]
        if word in DOUBLE:
            double_count += answer
        else:
            single_count += answer
    return double_count, single_count


def run_one_trial_recency(associations, function, lexicon):
    double_count = 0
    single_count = 0
    for word in GOLD:
        options = [GOLD[word][-1]]
        answer = run_one_word(lexicon, associations, function, word, options)[0]
        if word in DOUBLE:
            double_count += answer
        else:
            single_count += answer
    return double_count, single_count


def run_one_trial_both(associations, function, lexicon):
    double_count_prim = 0
    double_count_rec = 0
    single_count = 0
    for word in GOLD:
        options = []
        options.extend(GOLD[word])
        answer = run_one_word(lexicon, associations, function, word, options)
        if word in DOUBLE:
            if answer[1] == 0:
                double_count_prim += answer[0]
            else:
                double_count_rec += answer[0]
        else:
            single_count += answer[0]
    return double_count_prim, double_count_rec, single_count


def initialize_arrays(count):
    arrays = []
    for i in range(7):
        arrays.append(np.zeros(count, float))
    return arrays


def run_one_exp(model, exp, data, memory_size, count=1000):#300):
    print(model)
    association, meanings, lexicon = {}, [], {}
    learning_model = []
    experiment = []
    condition = []
    word_type = []
    answer = []
    stddev = []
    values = []
    memory_sizes = []
    all_runs = initialize_arrays(count)

    for i in range(count):
        current_memory_size = max(0, round(np.random.normal(memory_size, 1)))
        if model == "xsit":
            association, meanings, lexicon = online_xsit(buffer_size=current_memory_size, learning_data_path=data, tau=0.78)
            lexicon = make_lexicon(meanings, association, lexicon)
        if model == "pursuit":
            association, meanings, lexicon = pursuit(learning_data_path=data, buffer_size=current_memory_size)
            ms = [[], []]
            for m_i in range(len(meanings)):
                if '1' in meanings[m_i]:
                    ms[0].append(m_i)
                elif '2' in meanings[m_i]:
                    ms[1].append(m_i)

        d_p, s_p = run_one_trial_primacy(association, mc(meanings), lexicon)
        d_r, s_r = run_one_trial_recency(association, mc(meanings), lexicon)
        d_b_p, d_b_r, s_b = run_one_trial_both(association, mc(meanings), lexicon)
        single_participant = [s_p, s_r, s_b, d_p, d_r, d_b_p, d_b_r]
        for k in range(7):
            all_runs[k][i] = single_participant[k]
    for k in range(7):
        stddev.append(all_runs[k].std())
        values.append(all_runs[k].mean())
    learning_model.extend([model] * 7)
    experiment.extend([exp] * 7)
    condition.extend(["primacy", "recency", "both"] * 2)
    condition.append("both")
    word_type.extend(["singleton"] * 3)
    word_type.extend(["homophone"] * 4)
    answer.extend(["singleton", "singleton", "singleton", "first meaning", "second meaning", "first meaning", "second meaning"])
    memory_sizes.extend([memory_size] * 7)
    return learning_model, experiment, condition, word_type, answer, values, stddev, memory_sizes


EXPERIMENTS = [DATA1, DATA2]


def run_all(a):
    results = pd.DataFrame([[]]).drop(0)
    all_experiments = [[], [], [], [], [], [], [], []]

    for model in ["pursuit", "xsit"]:
        for e in range(2):
            data = EXPERIMENTS[e]
            one_experiment = run_one_exp(model, e+1, data, a)
            for i in range(8):
                all_experiments[i].extend(one_experiment[i])

    learning_model, experiment, condition, word_type, answer, values, stddev, memory_sizes = all_experiments
    results["model"] = learning_model
    results["experiment"] = experiment
    results["condition"] = condition
    results["word_type"] = word_type
    results["referent"] = answer
    results["accuracy"] = np.array(values) / 6
    results["stddev"] = np.array(stddev) / 6
    results["stderr"] = np.array(stddev) / (sqrt(48*10))
    results["buffer size"] = memory_sizes
    results.to_csv("./results/yurovsky_yu_results.csv", index=False)
    return results


def main():
    # Run with memory buffer size = 10
    run_all(10)


if __name__ == "__main__":
    main()
