import numpy as np
from random import choice
from lru_cache import LRUCache

from library import eval_model, parse_input_data


"""
Data structure: 
    Associations is a dictionary with key: word <str>
                                      value: array of association values <numpy.array<float>>
    Meanings is an array of strings, the index of the meaning corresponds to the
        index of the value
    This data structure allows for a sparsely populated matrix
"""


def ignore_learned(lex, w_u, m_u):
    """
    Don't want to include words that are already learned and learned object is in the utterance

    :param lex: {<str>: [<str>]} {word1:[meaning1, meaning2], ...}
    :param w_u: [<str>] list of words in utterance
    :param m_u: [<str>] list of objects / meanings in utterance
    :return:
    """
    n_w_u = []
    for w in w_u:
        include = True
        if w in lex:
            for m in lex[w]:
                if m in m_u:
                    include = False
        if include:
            n_w_u.append(w)
    return n_w_u


def remove_from_queue(q, associations, meanings, lexicon, threshold, beta, smoothing):
    """
    When it goes out of memory

    Remove from the memory buffer and from the association matrix
    Also, this is an opportunity to add to the lexicon
    This is important, because the key point of xsit is that the lexicon
    is not created until the very end.

    :param q: LRUCache memory buffer, limited number of words the model can learn
                concurrently
    :param associations: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param meanings: [<str>] list of objects that are possible hypotheses
    :param lexicon: {<str>: [<str>]} {word1:[meaning1, meaning2], ...}
    :param threshold: <float> also known as tau, the threshold value os association to be
                        learned
    :return: new memory buffer and association matrix
    """
    word = q.get()
    m = get_best_meaning(word, meanings, associations, threshold, beta, smoothing)
    if m:
        lexicon = add_to_lexicon(word, m, lexicon)
    associations.pop(word, None)
    return q, associations, lexicon


def check_buffer(associations, w_u, memory_buffer, meanings, lexicon, threshold, beta, smoothing):
    """
    Check to see if the memory buffer is full

    :param associations: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param w_u: [<str>] list of words in utterance
    :param memory_buffer: Queue<str> memory buffer, limited number of words the model can learn
                            concurrently
    :param meanings: [<str>] list of objects that are possible hypotheses
    :param lexicon: {<str>: [<str>]} {word1:[meaning1, meaning2], ...}
    :param threshold: <float> also known as tau, the threshold value os association to be
                        learned
    :return: modified memory_buffer and associations
    """
    for word in w_u:
        if word not in associations:
            if memory_buffer.full():
                memory_buffer, associations, lexicon = remove_from_queue(memory_buffer, associations, meanings, lexicon,
                                                                         threshold, beta, smoothing)
            memory_buffer.put(word)
        else:
            memory_buffer.reshuffle(word)
    return memory_buffer, associations, lexicon


def add_novel_words_meanings(m_all, assoc, w_u, m_u, beta):
    """
    Returns new associations with added words and meanings

    Initialize all P(new_word|meaning) = 0, P(word|new_meaning) = 0,
    and all P(word_in_utterance|meaning_in_utterance) initialized to 1/beta
    if it doesn't already have a value

    :param m_all: [<str>] list of all meanings
    :param assoc: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param w_u: [<str>] set of words in situation
    :param m_u: [<str>] set of meanings in situation
    """
    for w in w_u:
        if w not in assoc:
            assoc[w] = np.zeros(len(m_all))
    for m in m_u:
        if m not in m_all:
            m_all.append(m)
            for w in assoc:
                assoc[w] = np.append(assoc[w], 0.0)
    for w in w_u:
        for m in m_u:
            meaning_i = m_all.index(m)
            if assoc[w][meaning_i] == 0:
                assoc[w][meaning_i] = 1.0/beta


def modified_attention(w_u, m_u, meanings, associations):
    """
    Modifies the current_lexicon P(w|m) instead of P(m|w) aka A(w,m) in the paper

    Finds the new non-normalized P value for a single hypothesis
    Mutates the current_lexicon

    Alignment(w, m) = Alignment(w, m) + 1

    :param w_u: [<str>] set of words in utterance
    :param m_u: [<str>] set of meanings in utterance
    :param meanings: [<str>] list of meanings
    :param associations: Dictionary
    """
    for word in w_u:
        all_hyp = associations[word]
        total_attention = 0
        for meaning in m_u:
            total_attention += all_hyp[meanings.index(meaning)]
        for meaning in m_u:
            meaning_i = meanings.index(meaning)
            new_attention = all_hyp[meaning_i] / total_attention
            associations[word][meaning_i] = all_hyp[meaning_i] + new_attention


def get_best_meaning(word, meanings, assoc, threshold, beta, smoothing):
    """
    Gets the best meaning associated with a word

    :param word: <str> the word
    :param meanings: [<str>] list of objects encountered
    :param assoc: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param threshold: <float> tau, a parameter that determines when a word is learned
    :return: The best meaning
    """
    if word in assoc:
        new_assoc = (assoc[word] + smoothing) / (assoc[word].sum() + (beta * smoothing))
        a_w = new_assoc.max()
        if a_w > threshold:
            # If there are multiple meanings with max association, randomly sample
            max_meanings = []
            for meaning in meanings:
                meaning_i = meanings.index(meaning)
                if new_assoc[meaning_i] == a_w:
                    max_meanings.append(meaning_i)
            max_index = choice(max_meanings)
            best_meaning = meanings[max_index]
            return best_meaning
    return None


def add_to_lexicon(word, meaning, lexicon):
    """
    Add a word, meaning pair to the lexicon

    :param word: <str> word
    :param meaning: <str> corresponding object
    :param lexicon: {<str>: [<str>]} {word1:[meaning1, meaning2], ...}
    :return: Nothing, mutates lexicon
    """
    if word not in lexicon:
        lexicon[word] = []
    lexicon[word].append(meaning)
    return lexicon


def make_lexicon(meanings, assoc, lex, threshold=0.7, beta=100, smoothing=0.01):
    """
    Returns a lexicon made from the association table if >threshold

    Choose the meaning with the highest association score, and compare to threshold.
    If it's greater, add to lexicon.
    :param meanings: [<str>] list of objects that are possible hypotheses
    :param assoc: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param lex: {<str>: [<str>]} {word1:[meaning1, meaning2], ...}
    :param threshold: <float> also known as tau, the threshold value os association to be
                        learned
    :return: dict, lexicon
    """
    new_lex = {}
    for word in lex:
        new_lex[word] = lex[word]
    for word in assoc:
        best_meaning = get_best_meaning(word, meanings, assoc, threshold, beta, smoothing)
        if best_meaning is not None:
            new_lex = add_to_lexicon(word, best_meaning, new_lex)
    return new_lex


def online_xsit(tau=0.77, buffer_size=100, learning_data_path=None, beta=100, smoothing=0.01):
    """
    Run xsit on the PARSED_INPUT

    :param tau: float, threshold for lexicon
    :param buffer_size: int, number of words that a child can be learning
    :param learning_data_path: str, path to learning data
    :return:
    """
    associations = {}
    meanings = []
    lexicon = {}
    memory_buffer = LRUCache(buffer_size)

    if learning_data_path:
        parsed_input = parse_input_data(learning_data_path)

        for (w_u, m_u) in parsed_input:
            memory_buffer, associations, lexicon = check_buffer(associations, w_u, memory_buffer, meanings, lexicon, tau, beta, smoothing)
            add_novel_words_meanings(meanings, associations, w_u, m_u, beta)
            modified_attention(w_u, m_u, meanings, associations)
    return associations, meanings, lexicon


def online_xsit_utterance(associations, meanings, lexicon, memory_buffer, w_u, m_u, tau=0.77, beta=100, smoothing=0.01):
    """
    Run xsit on a single utterance

    Allows for experiments in which the model must learn and is tested on their lexicon

    :param associations: {<str>: <np.array<float>>}
    :param meanings: [<str>]
    :param lexicon: {<str>:[<str>]}
    :param memory_buffer: Queue current memory status
    :param w_u: [<str>] words present in utterance
    :param m_u: [<str>] objects present in utterance
    :param beta: float, beta
    :param tau: float, threshold for lexicon
    :param smoothing: float, lambda parameter for smoothing
    :return: associations {<str>: <np.array<float>>}, meanings [<str>], lexicon {<str>:[<str>]},
            memory_buffer Queue
    """
    memory_buffer, associations, lexicon = check_buffer(associations, w_u, memory_buffer, meanings, lexicon, tau,
                                                           beta, smoothing)
    add_novel_words_meanings(meanings, associations, w_u, m_u, beta)
    modified_attention(w_u, m_u, meanings, associations)
    return associations, meanings, lexicon, memory_buffer


def run_xsit(tau=0.77):
    """
    Runs xsit

    :param tau: float, threshold value
    """
    associations, meanings, lexicon = online_xsit(tau=tau, buffer_size=1000, learning_data_path="./data/train.txt")
    lexicon = make_lexicon(meanings, associations, lexicon, tau)

    lexicon_list = []
    for word in lexicon:
        print(word, lexicon[word])
        for meaning in lexicon[word]:
            lexicon_list.append((word, meaning))
    p, r, f = eval_model(lexicon_list)
    print(tau, "\n", "p: ", p, "\n", "r: ", r, "\n", "f: ", f)


if __name__ == "__main__":
    run_xsit()

