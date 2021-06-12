import pandas as pd
import numpy as np

from memory_buffer import MemoryBuffer
from library import parse_input_data, extract_golden_standard
from memory_bound_pursuit import pursuit_utterance, online_pursuit_mc
from memory_bound_xsit import online_xsit_utterance, make_lexicon


RESULT = ["learningModel", "subject", "word", "instance", "correct", "correctLast"]
gold = extract_golden_standard("data/trueswell/gold.txt")
GOLD_TRUESWELL = {}
for (word, meaning) in gold:
    GOLD_TRUESWELL[word] = meaning

TRAIN_TRUESWELL = parse_input_data("data/trueswell/training.txt")


def run_exp(count=200):
    results = pd.DataFrame([[]]).drop(0)
    all_runs = []
    for _ in RESULT:
        all_runs.append([])
    for model in ["pursuit", "xsit"]:
        for i in range(count):
            association = {}
            meanings = []
            lexicon = {}
            current_memory_size = max(1, round(np.random.normal(10, 1)))
            memory_buffer = MemoryBuffer(current_memory_size)
            instance = 0
            correct_last = {}
            for word in GOLD_TRUESWELL:
                correct_last[word] = False
            for [w_u, m_u] in TRAIN_TRUESWELL:
                word = w_u[0]
                if word == "heek":
                    instance += 1
                if model == "xsit":
                    association, meanings, lexicon, memory_buffer = online_xsit_utterance(association, meanings,
                                                                                          lexicon, memory_buffer, w_u,
                                                                                          m_u, tau=0.77)
                    new_lexicon = make_lexicon(meanings, association, lexicon, 0.77)
                    simulated_answer = online_pursuit_mc(word, m_u, new_lexicon, meanings, association)
                else:
                    association, meanings, lexicon, memory_buffer = pursuit_utterance(association, meanings, lexicon,
                                                                                      memory_buffer, w_u, m_u)
                    simulated_answer = online_pursuit_mc(word, m_u, lexicon, meanings, association)
                if word in GOLD_TRUESWELL:
                    correct = simulated_answer == GOLD_TRUESWELL[word]

                    all_runs[RESULT.index("learningModel")].append(model)
                    all_runs[RESULT.index("subject")].append(i+1)
                    all_runs[RESULT.index("word")].append(word)
                    all_runs[RESULT.index("instance")].append(instance)
                    all_runs[RESULT.index("correct")].append(1 if correct else 0)
                    all_runs[RESULT.index("correctLast")].append(1 if correct_last[word] else 0)
                    correct_last[word] = correct
    for item in RESULT:
        results[item] = all_runs[RESULT.index(item)]
    results.to_csv("results/trueswell_results.csv", index=False)
    print(results)
    return results


def main():
    run_exp()


if __name__ == "__main__":
    main()
