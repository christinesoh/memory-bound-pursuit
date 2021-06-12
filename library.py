def extract_words_meanings(situation):
    """
    Given an utterance, extract the words and objects present

    :param situation: the 2 lines in the training file
    :return: (w_u: list<str>, m_u: list<str>)
    """
    words_str, meanings_str = situation
    w_u = words_str.split()
    m_u = meanings_str.split()
    return w_u, m_u


def parse_input_data(training_data_path):
    """
    Parse the input data

    :param training_data_path: str, path to training data
    :return: list [(w_u_1, m_u_1), (w_u_2, m_u_2), etc] list of word, meaning
        lists utterances
    """
    parsed = []
    with open(training_data_path) as file:
        sit = []
        for line in file:
            if line == '\n':
                words, meanings = extract_words_meanings(sit)
                parsed.append((words, meanings))
                sit = []
            else:
                sit.append(line)
    return parsed


# Get evaluation metrics


def extract_golden_standard(path_to_standard):
    """
    Extracts the golden standard associations into a dict

    :param path_to_standard: str, path to the golden standard associations
    :return: list<(word<str>, meaning<str>)> tuples
    """
    lexicon = set()
    with open(path_to_standard) as f:
        for line in f:
            word, meaning = line.split()
            lexicon.add((word, meaning))
    return lexicon


def eval_model(lexicon):
    """
    Get back the evaluation metrics on a learned lexicon

    :param lexicon: list<(word<str>, meaning<str>)> of word, meaning tuples
    :return: precision, recall, f-score
    """
    lexicon_set = set(lexicon)
    correct = len(GOLDEN_STANDARD.intersection(lexicon_set))
    p = float(correct) / len(lexicon)
    r = float(correct) / len(GOLDEN_STANDARD)
    f = (2 * p * r) / (p + r)
    return p, r, f


def eval_model_gold(lexicon, gold):
    """
    Get back the evaluation metrics on a learned lexicon

    :param lexicon: list<(word<str>, meaning<str>)> of learned word, meaning tuples
    :param gold: list<(word<str>, meaning<str>)> of correct word, meaning tuples
    :return: precision, recall, f-score
    """
    lexicon_set = set(lexicon)
    golden_standard = extract_golden_standard(gold)
    correct = len(golden_standard.intersection(lexicon_set))
    p = float(correct) / len(lexicon)
    r = float(correct) / len(golden_standard)
    f = (2 * p * r) / (p + r)
    return p, r, f
