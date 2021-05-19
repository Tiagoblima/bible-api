import math

from nltk import RegexpTokenizer
import pandas as pd
import re
import matplotlib.pyplot as plt

from stemming.stemming import AutoStem


def calculate_freq(count_dic):
    freq_dic = {}
    total_count = sum(count_dic.values())

    for key, count in zip(count_dic.keys(), count_dic.values()):
        freq_dic[key] = count / total_count

    return freq_dic


def format_result(result):
    x, y = [], []
    for res in result:
        y.append(res[0])
        x.append('{0:.2f}%'.format(res[1]))
    return x, y


def draw_plot(params):
    x = params['x']
    y = params['y']

    plt.plot(x, y)
    plt.title(params['title'])
    plt.xticks(rotation=45)
    plt.xlabel(params['x_label'])
    plt.ylabel(params['y_label'])
    plt.figure(dpi=300)
    plt.show()


def stem_words(text, label):
    stem = AutoStem(text)
    stem.freq_counter()
    stem.stem_words()
    data = {label: list(filter(lambda x: type(x) == str, stem.select_stem()))}
    return data


def text_prep(text):
    """
    Remove partes indesejadas como números e palavras na stop_list. Além disso adicionar # ao final da
    palavra a fim de facilitar no stems de uma única letra
    :param text:
    :return:
    """
    text = list(filter(lambda x: type(x) == str, text))
    tokenizer = RegexpTokenizer(r'\w+', flags=re.UNICODE)
    tokens = tokenizer.tokenize(' '.join(text).lower())

    new_tokens = []
    w_freq = word_count(' '.join(tokens), set(tokens))

    stop_list = [tup[0] for tup in w_freq[:300]]
    for token in tokens:
        if token not in stop_list:
            token = ''.join([letter for letter in token if not letter.isdigit()])
            new_token = token + '#'
            new_tokens.append(new_token)

    return ' '.join(new_tokens)


def letter_count(token, count_dic):
    for let in token[:-1]:

        try:
            freq = count_dic.get(let.lower())
            freq += 1
            count_dic[let.lower()] = freq
        except TypeError:
            freq = 1
            count_dic[let.lower()] = freq
    return count_dic


def suffix_count(token, count_dic):
    for size in range(1, 8):

        if len(token) > size + 2:
            suffix = token[-size:-1]

            try:
                if suffix:
                    freq = count_dic.get(suffix)
                    freq += 1
                    count_dic[suffix] = freq
            except TypeError:
                freq = 1
                count_dic[suffix] = freq

    return count_dic


def word_count(text, words):
    word_freq = {}
    for word in words:
        counting = len(list(re.findall(word, text)))
        word_freq[word] = counting

    return sorted(word_freq.items(), key=lambda tup: tup[1], reverse=True)


def count_words(text, threshold=100):
    """
    Retorna as palavras mais frequentes limitadas pelo threshold que
    é 100 por default.
    """
    word_freq = {}.fromkeys(set(text.split(' ')))
    text = text.split(' ')
    for word in text:
        try:
            word_freq[word] += 1
        except TypeError:
            word_freq[word] = 1

    word_freq.pop('')

    return sorted(word_freq.items(), key=lambda kv: kv[1], reverse=True)[:threshold]


def stem_text(text, stems):
    text = ' '.join(list(filter(lambda x: type(x) == str, text))).lower()

    for stem in stems:
        words = re.findall(stem + r'\w+', text)
        for word in words:
            text = text.replace(word, stem)
    return text


def to_dic(list_tup):
    suffixes = []
    cohere = []

    for tup in list_tup:
        suffixes.append(tup[0])
        cohere.append(tup[1])

    suffix_dic = {
        'suffix': suffixes,
        'coherence': cohere
    }

    return suffix_dic


def coherence(suffix_data, letter_freq):
    suffixes = suffix_data[0]
    s_freq = suffix_data[1]
    ls_freq = 1
    for let in suffixes:
        l_freq = letter_freq.get(let.lower())
        ls_freq *= l_freq
    coh = s_freq * math.log((s_freq / ls_freq), 10)

    return coh


def save_stemming(stemmed_data, path):
    stems = []
    words = []
    sufix = []
    s_freq = []
    # (stem, word, sufix, freq)
    for list_tup in stemmed_data['signature'].values():

        for el in list_tup:
            stems.append(el[0])
            words.append(el[1])
            sufix.append(el[2])
            s_freq.append(el[3])

    df = pd.DataFrame({
        'stems.csv': stems,
        'words': words,
        'suffix': sufix,
        'freq': s_freq
    })

    df.to_csv(path, index=False)
