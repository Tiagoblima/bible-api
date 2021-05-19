import re
import sys

from bs4 import BeautifulSoup
from nltk import RegexpTokenizer
from numpy.random.mtrand import shuffle

from stemming.util import text_prep, coherence


class AutoStem:
    path_dir = ''
    raw_text = ''

    def __init__(self, text):

        """
        Initialize the required variables
        :param text:
        """

        self.data = {
            'letter': {},
            'suffix': {},
        }

        self.candidates = {}
        self.suffixes_stem = {}
        self.suffix_coh = set()

        self.raw_text = text_prep(text)

    def get_suffix_freq(self):
        suffix_dic = self.data['suffix']
        return suffix_dic

    def get_letter_freq(self):
        letter_dic = self.data['letter']
        return letter_dic

    def get_text(self):
        return self.raw_text

    def freq_counter(self):

        letter_freq = {}
        suffix_freq = {}

        tokens = self.raw_text.split()

        for token in tokens:

            for let in token[:-1]:

                try:
                    freq = letter_freq.get(let.lower())
                    freq += 1
                    letter_freq[let.lower()] = freq

                except TypeError:

                    freq = 1
                    letter_freq[let.lower()] = freq

                for size in range(1, 8):

                    if len(token) > size:
                        suffix = token[-size:-1]

                        try:
                            freq = suffix_freq.get(suffix)
                            freq += 1
                            suffix_freq[suffix] = freq
                        except TypeError:
                            freq = 1
                            suffix_freq[suffix] = freq

        total_count = sum(letter_freq.values())

        for key, count in zip(letter_freq.keys(), letter_freq.values()):
            letter_freq[key] = count / total_count

        total_count = sum(suffix_freq.values())

        for key, count in zip(suffix_freq.keys(), suffix_freq.values()):
            suffix_freq[key] = count / total_count
        suffix_freq.pop('')
        self.data['letter'] = letter_freq
        self.data['suffix'] = suffix_freq

        return suffix_freq

    def select_stem(self, threshold=100):
        """
        Select the best suffixes
        Key args:

        threshold defines the number of suffix evaluated
        """
        selected = []
        top_suffixes = [tup[0] for tup in self.suffix_coh]

        for suffix in top_suffixes[:threshold]:
            stems = self.suffixes_stem[suffix]
            if len(set(stems)) >= 2:
                for stem in stems:
                    suffix = self.candidates[stem]
                    if len(set(suffix)) <= 5:
                        tokenizer = RegexpTokenizer(r'\w+', flags=re.UNICODE)
                        stem = ' '.join(tokenizer.tokenize(stem.lower()))
                        selected.append(stem)
        return selected

    def stem_words(self):
        """
            Stem the words based on its coherence
        """

        suffix_freq = self.data['suffix']

        suffixes = list(suffix_freq.keys())

        temp = 0
        print("Stemming: ")
        total = len(suffixes)
        print('Loading: ', end='')

        print(suffixes[:10])
        for suffix in suffixes:
            self.suffixes_stem[suffix] = set()
            sys.stdout.write('\r' + 'Loading: {:.2f}'.format(temp / total * 100) + '%')
            sys.stdout.flush()

            search = re.findall(r'\w+' + str(suffix) + '#', self.raw_text)

            freq = suffix_freq[suffix]
            coh = coherence((suffix, freq), self.data['letter'])
            self.suffix_coh.add((suffix, coh))

            for word in set(search):
                stem = word.lower().replace(suffix + '#', '')
                self.suffixes_stem[suffix].add(stem)  # Store the stems associated with the prefix
                try:
                    self.candidates[stem].add(suffix)
                except KeyError:
                    self.candidates[stem] = set(suffix)  # Store the suffix associated with the stem

            temp += 1

        self.suffix_coh = set(sorted(self.suffix_coh, key=lambda tup: tup[1], reverse=True))

    def get_data(self):
        return self.data

    def signatures(self):
        return self.candidates
