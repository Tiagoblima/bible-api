import codecs
import json
import random
import re
import sqlite3
import stat
import sys
import time
import os
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup
import numpy as np

from textprep.params import TRAINING_TEXT, VALIDATION_TEXT, TESTING_TEXT, PATH_TO_TEXT_FILES

DIC_PATH = 'Resources/guarani_dict/'
pd.set_option('display.max_colwidth', None)


def save_pairs(self, file_type='.txt', dir_to_save=r'Resources/pairs/'):
    print('\nSaving pairs: ')
    print('Progress: #', end='')
    for key, text in zip(self.data_pairs.keys(), self.data_pairs.values()):

        path = dir_to_save + key + file_type
        print('#', end='')

        file = open(path, 'w', encoding='utf-8')

        for line in text:
            file.write(line)

        file.close()


def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = np.unicode(text, 'utf-8')
    except (TypeError, NameError):  # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")

    return str(text)


def get_dictionary(save=True):
    print('Obtaining dictionary:')
    dic = {}
    with codecs.open(DIC_PATH, encoding='windows-1252', errors='replace') as file:
        text = file.read()

    text_dic = re.findall(r'<p>.*</p>', text)
    for line in text_dic:

        for portuguese, guarani in zip(re.findall(r'<p>.*</b>', line), re.findall(r'</b>\s.*</p>', line)):
            portuguese = re.sub(r'<.*?>', '', portuguese)
            guarani = re.sub(r'<.*?>', '', guarani)
            dic[portuguese.lower()] = guarani.lower()

    if save:

        joined = []
        for guarani, portuguese in zip(list(dic.values()), list(dic.keys())):
            joined.append(guarani + '\t' + portuguese)

        df = pd.DataFrame({joined[0]: joined[1:]})
        df.to_csv('guarani-portugues.csv', index=False)

    return dic


def scrap_dic():
    print('Obtaining dictionary:')
    dic = {}
    w_type = 'unknown'
    for achieve in os.listdir(DIC_PATH):

        with open(DIC_PATH + achieve, encoding='utf-8', errors='replace') as file:
            soup = BeautifulSoup(file, 'html.parser')

            soup_dic = soup.find('div', id="dic").extract()
            soup_dic = soup_dic.find_all('p')

            for i, el in list(enumerate(soup_dic)):

                try:
                    for font in el.find_all('font'):
                        font.unwrap()
                except AttributeError:
                    pass

                tag_type = type(el.find('strong'))

                for word in el.find_all('strong'):
                    if len(word.get_text(strip=True)) == 0:
                        word.extract()
                    else:
                        str_word = ''.join(word.text.strip())
                        description = []

                        for sibiling in el.find('strong').next_siblings:
                            if type(sibiling) is not tag_type:
                                description.append(sibiling)
                            else:
                                break

                        description = ''.join(description)
                        types = [r'\sadj\.', r'\sadv\.', r'\sconj\.', r'\scoord\.', r'\sinterrog\.', r'\slit\.',
                                 r'\snomin\.', r'pl\.', r'\sposp\.', r'\sposs\.', r'\spref\.', r'\spron\.', r'\ss\.',
                                 r'\ssing\.', r'\ssubord\.',
                                 r'\ssuf\.', r'\sv\.', r'\st\.', r'\si[.]', r'\spron[.]', r'\spref[.]']

                        types = [match for w_type in types for match in re.findall(w_type, description)]
                        if len(types) is not 0:
                            print('type:', )
                            w_type = ''.join(r'\s'.join(types).split('\n')).rstrip()
                        for tp in w_type.split(r'\s'):
                            w_type = ''.join(tp.split(r'\s')).strip()

                            dic[w_type].append(''.join(str_word.split('\n')))

                        print("Word: ", str_word)
                        print("Word type: ", w_type)
                        print("Description: ", ''.join(description))

                    word.extract()
                    print('-' * 40)

    dict_df = pd.DataFrame.from_dict(dic, orient='index').fillna('')
    dict_df = dict_df.T.drop(columns=['unknown'])

    dict_df.to_csv('words_dict.csv', index=False)


def animated_loading():
    chars = r"/—\|"
    for char in chars:
        sys.stdout.write(char)
        time.sleep(.1)
        sys.stdout.flush()


def from_text_to_file(text, path=None):
    assert path is not None, "Path cannot be NoneType."
    with open(path, 'w', encoding='utf8') as file:
        for verse in text:
            verse = verse.strip().replace('\"', '')

            file.write(verse + '\n')
        file.close()


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = w.strip()

    # creating a space between a word and the punctuation following it eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping
    # -punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^\w?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.

    return w


def from_json_to_csv(data, filename='file.csv'):
    columns = ["Book", "Chapter", "Verse", "Scripture"]
    bible = dict.fromkeys(columns)

    for column in columns:
        bible[column] = []

    for idxB, book in enumerate(data):

        for idxC, chapter in enumerate(book['chapters']):

            for idxV, verse in enumerate(chapter):
                bible["Chapter"].append(idxC + 1)
                bible["Book"].append(idxB + 1)
                bible["Verse"].append(idxV + 1)
                bible["Scripture"].append(verse)

    df = pd.DataFrame().from_dict(bible, orient='index').T.dropna()

    df.to_csv(filename, index_label=False, columns=columns, index=False)
    return df


def convert_json_to_csv(base_dir='../Resources/json/'):
    for file in os.listdir(base_dir):
        data = json.load(open(base_dir + file, encoding='utf-8-sig'))
        yield from_json_to_csv(data, filename=base_dir + file.split('.')[0] + '.csv')


def delete_all_files(base_dir):
    for f in os.listdir(base_dir):
        os.chmod(os.path.join(base_dir, f), stat.S_IWRITE)
        os.remove(os.path.join(base_dir, f))


def normalize_strings(strings_list):
    """
        Strip extra space from strings and also change the reference to a standard format
        <sup>([][]-[])</sup>.
    """

    for i, string in enumerate(strings_list):
        new_string = string.strip()
        strings_list[i] = ' '.join(new_string.replace('\"', '').split())
        found_ref = re.findall(r'\(\s[0-9][0-9]*\s[-]\s[0-9][0-9]*\s\)', new_string)
        for ref_string in found_ref:
            new_ref = '<sup>' + ''.join(ref_string.split()) + '</sup>'
            strings_list[i] = strings_list[i].replace(ref_string, new_ref)
            print(i, strings_list[i])

    return strings_list


def list_to_pairs(scripture_1, scripture_2):
    lines = []
    for verse_1, verse_2 in zip(scripture_1, scripture_2):
        line = verse_1.strip().replace('\"', '') + '\t' + verse_2.strip().replace('\"', '') + '\n'
        lines.append((len(line), line))
    lines.sort(key=lambda tup: tup[0])

    return [line[1] for line in lines]


def generate_meta_info(path_to_books=os.curdir, path_to_save=os.curdir):
    version = "nvi"
    bible_meta_info = {
        "version": version
    }
    for filename in os.listdir():
        data = json.load(open(path_to_books + filename, 'r', encoding='utf-8'))
        chapter_info = {}
        for chap in data:
            chapter = list(chap.keys())[0]
            text = list(list(chap.values())[0].values())
            chapter_info[chapter] = {"chapter": int(chapter),
                                     "verses": len(text),
                                     "text": text}
        bible_meta_info[filename.split('.')[0]] = chapter_info

    json.dump(bible_meta_info, open(path_to_save + 'meta-info.json', 'w'), indent=2)


def selecting_books(path_to_dir=None, column_name=None, number=0):
    if path_to_dir is None:
        path_to_dir = os.curdir
    assert column_name is not None, "You must provide a column name"
    for filename in os.listdir(path_to_dir):
        df = pd.read_csv(path_to_dir + filename, encoding='utf-8')
        df = df[df[column_name] >= number]
        df.to_csv(path_to_dir + filename, index=False, index_label=False)


def sqlite_to_csv(db_dir=None, output_dir=None, table=None, query=None, save_to_csv=False):
    """
        Converts a directory of sqlite files into a csv one.
        return: the lists of saved data-sets
    """

    if db_dir is None:
        db_dir = os.curdir
    if output_dir is None:
        output_dir = os.curdir
    if query is None:
        query = 'SELECT * FROM '

    assert type(table) is str, "Table name must be a string, got {} instead".format(table)
    dfs = []
    for filename in os.listdir(db_dir):

        conn = sqlite3.connect(os.path.join(db_dir,filename))
        df = pd.read_sql_query(query + table, conn)
        if save_to_csv:
            df.to_csv(os.path.join(output_dir, filename.split()[0]) + '.csv', index=False, index_label=False)
        dfs.append(df)
    return dfs


def train_test_split(test_size=0.1, val_size=.08):
    """
        Splits the text file into other two (training and validation)
        test_size=percentage of the test size
    """

    try:
        os.mkdir(TRAINING_TEXT)
        os.mkdir(TESTING_TEXT)
        os.mkdir(VALIDATION_TEXT)
    except OSError:
        pass

    text_files = [file for file in os.listdir(PATH_TO_TEXT_FILES) if len(file.split('.')) == 2]

    file = open(os.path.join(PATH_TO_TEXT_FILES, text_files[0]), encoding='utf-8')
    total_examples = len(file.readlines())
    num_test_examples = int(total_examples * test_size)
    num_val_examples = int(total_examples * val_size)
    test_indexes = random.choices(range(total_examples), k=num_test_examples)
    val_indexes = random.choices(np.delete(range(total_examples), test_indexes), k=num_val_examples)

    for filename in text_files:
        file = open(os.path.join(PATH_TO_TEXT_FILES, filename), encoding='utf-8')
        text_list = file.readlines()

        print("{} Training {}, test {},  validation {} split".format(filename.split('.')[0],
                                                                     total_examples - num_test_examples - num_val_examples,
                                                                     num_test_examples, num_val_examples
                                                                     ))
        # pdb.set_trace()

        training_file = open(os.path.join(TRAINING_TEXT, filename), 'w', encoding='utf8')
        testing_file = open(os.path.join(TESTING_TEXT, filename), 'w', encoding='utf8')
        val_file = open(os.path.join(VALIDATION_TEXT, filename), 'w', encoding='utf8')

        dataset = {"test": [],
                   "train": [],
                   "val": []}
        for i, line in enumerate(text_list):
            if i in test_indexes:
                dataset["test"].append(line)
                testing_file.write(line)
            elif i in val_indexes:
                dataset["val"].append(line)
                val_file.write(line)
            else:
                dataset["train"].append(line)
                training_file.write(line)

        testing_file.close()
        training_file.close()
        return dataset


def get_lang_id(lang):
    length = len(lang)
    return lang[0] + lang[int(length / 2)] + lang[-1]


def create_augmented_file(versions=None,
                          path=None,
                          aug_name='portugues',
                          to_txt=False):
    """
    Creates a text file of pair translation cross a list of versions
    """
    if versions is None:
        versions = []
    aug_text = []
    unique_versions = {}

    for filename in os.listdir(path):
        try:
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                if filename in versions:
                    aug_text.append(file.readlines())
                else:
                    unique_versions[filename] = file.readlines()
        except PermissionError:
            pass

    text_files = {}

    for unique_text, key in zip(unique_versions.values(), unique_versions.keys()):
        filename = key.split('.')[0] + '-' + aug_name

        text_files[filename] = []
        filename += '.txt'
        for i, u_text in enumerate(unique_text):
            for text in aug_text:
                line = preprocess_sentence(u_text.strip('\n')) + '\t' + preprocess_sentence(text[i]) + '\n'
                text_files[filename].append((line, len(line)))
        text_files[filename].sort(key=lambda tup: tup[1])

        lines = [line[0] for line in text_files[filename]]
        lines = pd.Series(lines).drop_duplicates().tolist()
        text_files[filename] = lines

        if to_txt:
            file = open(os.path.join(path, filename), 'w', encoding='utf8')
            for line in lines:
                file.write(line)

            file.close()
    return json.dumps(text_files)
