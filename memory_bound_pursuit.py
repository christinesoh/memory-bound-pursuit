from random import choice, choices
import numpy as np
from lru_cache import LRUCache

from library import parse_input_data


"""
Data structure: 
    Associations is a dictionary with key: word <str>
                                      value: array of association values <numpy.array<float>>
    Meanings is an array of strings, the index of the meaning corresponds to the
        index of the value
    This data structure allows for a sparsely populated matrix
"""


LEARNED = 100
NULL_HYPOTHESIS = -1


def add_novel_meanings(association_matrix, meanings, m_u):
    """
    A function that adds objects that have not yet been encountered to the association matrix

    :param association_matrix: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param meanings: [<str>] list of objects that are possible hypotheses
    :param m_u: [<str>] the list of objects present in the utterance
    :return: None, but mutates the association matrix
    """
    for meaning in m_u:
        # Check to see if the meaning is new
        if meaning not in meanings:
            meanings.append(meaning)
            for word in association_matrix:
                association_matrix[word] = np.append(association_matrix[word], NULL_HYPOTHESIS)


def online_me(lexicon, associations, meanings, m_u):
    """
    Mutual exclusivity for online pursuit

    returns the best possible object associated with a word; based on which meaning is least "tied" to another word

    :param lexicon: {<str>: [<str>]} {word1:[meaning1, meaning2], ...}
    :param associations: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param meanings: [<str>] list of objects that are possible hypotheses
    :param m_u: [<str>] the list of objects present in the utterance
    :return: int; index of the best meaning choice from m_u
    """
    # to keep track of the highest association for each meaning
    max_associations = []
    for m in m_u:
        a_max = -1
        for word in associations:
            a_m_w = associations[word][meanings.index(m)]
            if a_m_w > a_max:
                a_max = a_m_w
        for word in lexicon:
            if m in lexicon[word]:
                a_max = LEARNED
        max_associations.append(a_max)
    # get the minimum value of the max associations
    # of the possible meanings in the utterance
    min_val = min(max_associations)
    possible_meaning_indices = []
    for i in range(len(m_u)):
        if max_associations[i] == min_val:
            possible_meaning_indices.append(meanings.index(m_u[i]))
    return choice(possible_meaning_indices)


def get_meaning_i(associations, word):
    """
    Gets the best hypothesis for a word given the association values

    :param associations: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param word: <str> the word
    :return: index of the object that is most associated with the word
    """
    max_h_list = []
    max_association = associations[word].max()
    for i in range(len(associations[word])):
        if associations[word][i] == max_association:
            max_h_list.append(i)
    return choice(max_h_list)


def add_novel_word(lexicon, associations, meanings, word, m_u):
    """
    Add a word that has yet to be encountered to the association matrix

    :param lexicon: {<str>: [<str>]} {word1:[meaning1, meaning2], ...}
    :param associations: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param meanings: [<str>] list of objects that are possible hypotheses
    :param word: <str> the word
    :param m_u: [<str>] the list of objects present in the utterance
    :return: Nothing, mutates association matrix
    """
    meaning_i_from_me = online_me(lexicon, associations, meanings, m_u)
    associations[word] = np.zeros(len(meanings))
    for m in range(len(meanings)):
        associations[word][m] = NULL_HYPOTHESIS
    associations[word][meaning_i_from_me] = 0


def naive_update_association(associations, word, meaning):
    """
    Update association by adding 1 if the hypothesis is confirmed

    :param associations: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param word: <str> the word
    :param meaning: <str> the object in question
    :return: Nothing, mutates the association matrix
    """
    if associations[word][meaning] == NULL_HYPOTHESIS:
        associations[word][meaning] = 0
    else:
        associations[word][meaning] += 1


def update_word(associations, meanings, word, m_u):
    """
    Do a step of pursuit for a single word

    :param associations: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param meanings: [<str>] list of objects that are possible hypotheses
    :param word: <str> the word
    :param m_u: [<str>] the list of objects present in the utterance
    :return: Nothing, mutates association matrix
    """
    hypothesis = get_meaning_i(associations, word)
    if meanings[hypothesis] not in m_u:
        all_but = [a for a in m_u if a != meanings[hypothesis]]
        new_hyp = meanings.index(choice(all_but))
        naive_update_association(associations, word, new_hyp)
    else:  # if meanings[hypothesis] in m_u
        naive_update_association(associations, word, hypothesis)


def remove_from_queue(q, associations):
    """
    When it goes out of memory

    Remove from the memory buffer and from the association matrix

    :param q: MemoryBuffer, limited number of words the model can learn
                concurrently
    :param associations: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :return: new memory buffer and association matrix
    """
    word = q.get()
    associations.pop(word, None)
    return q, associations


def train_on_utterance(memory, associations, meanings, lexicon, w_u, m_u):
    """
    Training the pursuit model on a single utterance

    :param memory: Queue<str> memory buffer, limited number of words the model can learn
                    concurrently
    :param associations: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param meanings: [<str>] list of objects that are possible hypotheses
    :param lexicon: {<str>: [<str>]} {word1:[meaning1, meaning2], ...}
    :param w_u: [<str>] the list of words present in the utterance
    :param m_u: [<str>] the list of objects present in the utterance
    :return: the new memory queue buffer; mutates the association matrix, meaning array,
            and lexicon
    """
    add_novel_meanings(associations, meanings, m_u)
    for w in w_u:
        if w in lexicon:
            meaning_in_m_u = False
            for m in lexicon[w]:
                if m in m_u:
                    meaning_in_m_u = True
            if meaning_in_m_u:
                continue

        if w not in associations:
            if memory.full():
                memory, associations = remove_from_queue(memory, associations)
            memory.put(w)
            add_novel_word(lexicon, associations, meanings, w, m_u)
        else:
            update_word(associations, meanings, w, m_u)
    return memory


def online_pursuit_mc(word, options, lexicon, meanings, associations):
    """
    Function for doing a multiple choice question

    :param word: <str> the word
    :param options: [<str>] the objects presented as options
    :param lexicon: {<str>: [<str>]} {word1:[meaning1, meaning2], ...}
    :param meanings: [<str>] list of objects that are possible hypotheses
    :param associations: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :return: <str> the choice
    """
    # the word is in the lexicon and the learned meaning is in the options
    if word in lexicon:
        learned_meanings = set(options).intersection(set(lexicon[word]))
        if len(learned_meanings) > 0:
            return choice(list(learned_meanings))

    # if the word isn't in memory, select randomly
    if word not in associations:
        return choice(options)

    # otherwise: sample from the associations
    possible_associations = []
    for meaning in options:
        # No negative weights --> make the negative weight 0
        if associations[word][meanings.index(meaning)] == NULL_HYPOTHESIS:
            possible_associations.append(0)
        else:
            possible_associations.append(associations[word][meanings.index(meaning)])
    # choices doesn't like it if they're all zero, make them all weighted the same
    if sum(possible_associations) == 0.0:
        for i in range(len(possible_associations)):
            possible_associations[i] = 1.0
    return options[choices(range(len(possible_associations)), possible_associations)[0]]


def update_lexicon(associations, meanings, lexicon):
    """
    "Learn" words, move from association matrix to a learned lexicon

    :param associations: {<str>: <np.array<float>>} matrix of associations (keeps a tally)
    :param meanings: [<str>] list of objects that are possible hypotheses
    :param lexicon: {<str>: [<str>]} {word1:[meaning1, meaning2], ...}
    :return: lexicon {<str>: [<str>]}, associations {<str>: <np.array<float>>}
    """
    for word in associations:
        top_score = sorted(associations[word])[-1]
        closest_competitor_score = NULL_HYPOTHESIS
        if len(meanings) > 1:
            closest_competitor_score = sorted(associations[word])[-2]
        meaning_i = None
        if closest_competitor_score == NULL_HYPOTHESIS:
            closest_competitor_score = 0
        # Must be more than 2x score of closest competitor
        if top_score > 2 * closest_competitor_score + 1:
            meaning_i = np.where(associations[word] == top_score)[0][0]
        if meaning_i is not None:
            if word not in lexicon:
                lexicon[word] = []
            lexicon[word].append(meanings[meaning_i])
            associations[word][meaning_i] = NULL_HYPOTHESIS
    return lexicon, associations


def pursuit(learning_data_path, buffer_size=10):
    """
    Function that runs pursuit

    :param buffer_size: <int> the length of the memory buffer; i.e. how many words the model
                    can be learning concurrently
    :param learning_data_path: <str> path to the training data, in the form
            words
            objects

            separated by a line break
    :return: associations {<str>: <np.array<float>>}, meanings [<str>], lexicon {<str>:[<str>]}
    """
    associations = {}
    memory_buffer = LRUCache(buffer_size)
    meanings = []
    lexicon = {}

    parsed_input = parse_input_data(learning_data_path)
    for (w_u, m_u) in parsed_input:
        memory_buffer = train_on_utterance(memory_buffer, associations, meanings, lexicon, w_u, m_u)
        lexicon, associations = update_lexicon(associations, meanings, lexicon)
    return associations, meanings, lexicon


def pursuit_utterance(associations, meanings, lexicon, memory_buffer, w_u, m_u):
    """
    Function that runs pursuit on a single utterance

    Allows for probing

    :param associations: {<str>: <np.array<float>>}
    :param meanings: [<str>]
    :param lexicon: {<str>:[<str>]}
    :param memory_buffer: Queue current memory status
    :param w_u: [<str>] words present in utterance
    :param m_u: [<str>] objects present in utterance
    :return: associations {<str>: <np.array<float>>}, meanings [<str>], lexicon {<str>:[<str>]},
            memory_buffer Queue
    """
    memory_buffer = train_on_utterance(memory_buffer, associations, meanings, lexicon, w_u, m_u)
    lexicon, associations = update_lexicon(associations, meanings, lexicon)
    return associations, meanings, lexicon, memory_buffer

