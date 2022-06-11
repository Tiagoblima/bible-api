import datetime
import itertools
import json
import os
import random
import re
import sys
import time
from multiprocessing import Pool

import nltk
import numpy as np
import pandas as pd

from textprep.bible.util import find_joined_ref, get_references
from textprep.util.util import delete_all_files, from_text_to_file, list_to_pairs, preprocess_sentence, sqlite_to_csv
from textprep.util.util import unicode_to_ascii


class CleanUpText:

    def __init__(self, df):
        self.df = df

    def clean_tags(self):
        pass


patterns = [r'<sup>\(.*\)</sup>',
            r'<.*?>[;]*',
            r'\[[0-9][0-9]*-[0-9][0-9]*]',
            r'\(-\)',
            r'\[\d+-\d+\]\s']


# r'\s[,]\s[0-9]{1,3}[-][0-9]{1,3}', r'\s[;]\s', ',
def clean_df(df):
    version, df = df
    cleaners = [re.compile(pattern) for pattern in patterns]
    for cleaner in cleaners:
        df.replace(to_replace=cleaner, value=' ', regex=True, inplace=True)

    refs = get_references()
    for ref in refs:
        df.replace(to_replace=ref, value=' ', regex=True, inplace=True)

    return version, df


def generate_indexes(total_examples, test_per, val_per):
    num_test_examples = int(total_examples * test_per)
    num_val_examples = int(total_examples * val_per)
    print(total_examples)
    test_indexes = random.choices(range(total_examples), k=num_test_examples)
    val_indexes = random.choices(np.delete(range(total_examples), test_indexes), k=num_val_examples)
    print(num_val_examples, num_test_examples)
    train_indexes = random.choices(np.delete(range(total_examples),
                                             np.concatenate((test_indexes, val_indexes)), axis=0),
                                   k=num_val_examples)

    if np.any((np.array(train_indexes) > total_examples)):
        print("Invalid values generated.")
    if np.any((np.array(test_indexes) > total_examples)):
        print("Invalid values generated.")
    if np.any((np.array(val_indexes) > total_examples)):
        print("Invalid values generated.")
    return {"train": train_indexes, "test": test_indexes, "dev": val_indexes}


class Dataset:

    def __init__(self, df):
        self.df = df.dropna()
        self.n_gram = 1

    def create_dataset(self, to_split=False, type='simply'):
        dataset = {}
        if sys.intern(type) is sys.intern('simply'):
            if to_split:
                dataset = {}
            for key in self.df.columns:

                dataset[key] = self.text_split(self.df[key])

            else:
                dataset = self.df.to_dict(orient='list')

        elif sys.intern(type) is sys.intern('lang_cls'):

            json_data = json.loads(self.df.to_json(orient='records', indent=4))
            df = pd.concat([pd.DataFrame(list(i.items()), columns=['LANG', 'TEXT']) for i in json_data],
                           ignore_index=True).sample(frac=1).reset_index(drop=True)
            return df

        return dataset

    def text_split(self, test_per=0.8, val_per=0.1, indexes=None):

        if not indexes:

            indexes = generate_indexes(self.df.size, test_per, val_per)
            dataset = {}

            for key in indexes.keys():
                dataset[key] = self.df.iloc[indexes[key]].tolist()
        else:
            dataset = {}
            for key in indexes.keys():
                dataset[key] = self.df.iloc[indexes[key]].tolist()

        return dataset

    def get_text_only(self):
        return self.df.drop(columns=["Book", "Chapter", "Verse"])

    def text_augment(self, reference_versions=None,
                     to_split=True,
                     test_per=None,
                     val_per=None,
                     joined_name=None):
        if not joined_name:
            joined_name = 'joined_text'
        dataset = self.create_dataset()

        new_dataset = {}
        assert reference_versions is not None, "You need to provide versions to be the reference"
        ref_dfs = self.df[reference_versions]

        for version in self.df.columns:
            if version not in reference_versions:

                text_list = []
                for i, verse in enumerate(dataset[version]):

                    for pt_verse in ref_dfs.iloc[i].tolist():
                        line = preprocess_sentence(verse) + '\t' + preprocess_sentence(pt_verse) + '\n'
                        text_list.append(line)

                df = pd.DataFrame.from_dict({version + '-' + joined_name: text_list})
                new_indexes = df[version + '-' + joined_name].str.len().sort_values().index
                df.reindex(new_indexes)
                df1 = df.reindex(new_indexes)
                df = df1.reset_index(drop=True)

                if to_split:
                    indexes = generate_indexes(ref_dfs[reference_versions[0]].size, test_per, val_per)

                    new_dataset[version + '-' + joined_name] = self.text_split(df[version + '-joined_name'],
                                                                               indexes=indexes)
                    test_df = self.df[reference_versions + [version]].iloc[indexes['test']]
                    test_df.set_index(version, inplace=True)

                    new_dataset[version + '-' + joined_name]['test'] = list(test_df.T.to_dict(orient='list').items())
                else:
                    new_dataset[version + '-' + joined_name] = df.to_dict()

        return new_dataset

    def _to_n_gram(self, sentence):
        return list(map("".join, nltk.ngrams(sentence, self.n_gram, pad_left=True, pad_right=True, left_pad_symbol='_',
                                             right_pad_symbol='_')))

    def n_gram_dataset(self, n_gram):

        n_gram_dataset = {}

        for lang in self.df.columns:
            for n in n_gram:
                self.n_gram = n
                n_gram_dataset[lang] = self._to_n_gram(" ".join(self.df[lang]))

        return n_gram_dataset

    def create_pairs(self, args):

        lang_1 = args[0]
        lang_2 = args[1]

        scripture_1 = self.df[lang_1].tolist()
        scripture_2 = self.df[lang_2].tolist()

        pair_list = list_to_pairs(scripture_1, scripture_2)

        lang_pair = {lang_1 + '-' + lang_2: self.text_split(pair_list)}
        return lang_pair

    def to_lang_pairs(self):

        labels = self.df.columns
        p = Pool(3)
        return list(p.map(self.create_pairs, itertools.combinations(labels, 2)))


class TextPreprocess:
    dataframes = None

    root_dir = 'adasd'
    data_pairs = {}

    def __init__(self, dir_path,
                 output_dir=os.curdir,
                 from_sqlite=True,
                 clean_out_dir=False,
                 load_files=True):
        self.version_list = []
        self.output_dir = output_dir
        self.root_dir = dir_path

        if from_sqlite:
            self.dataframes = sqlite_to_csv(dir_path, table='verses')
            breakpoint()
        else:
            self.dataframes = []
            self.get_dataframes()

        if load_files:
            self.version_list = [unicode_to_ascii(version.split(' ')[0].lower())
                                 for version in os.listdir(dir_path)]

        self.aligned_df = pd.DataFrame(columns=["Book", "Chapter", "Verse"])

        if clean_out_dir:
            delete_all_files(output_dir)
        self.logs = {}

    def set_root_dir(self, dir_path):
        self.root_dir = dir_path
        self.dataframes = []
        self.version_list = os.listdir(dir_path)
        self.get_dataframes()

    def set_aligned_csv(self, df):
        self.aligned_df = df

    def get_aligned_csv(self):
        return self.aligned_df

    def get_prefix(self):
        return self.root_dir

    def get_dataset_list(self):
        return self.version_list

    def set_datasets(self, datasets):
        self.dataframes = datasets

    def get_dataframes(self):

        for name in os.listdir(self.root_dir):

            path = os.path.join(self.root_dir, name)
            df = pd.read_csv(path, encoding='utf-8')
            num_verses = df.shape[0]
            df = df.drop_duplicates(subset=['Scripture'])
            print(f"{name}- duplicates Deleted verses: ", num_verses - df.shape[0])
            try:
                self.dataframes.append((unicode_to_ascii(name.lower()), df
                                        ))

            except FileNotFoundError:
                return "The path " + path + " was not found."

        return self.dataframes

    def get_dataset(self, dataset_name):
        dataset = self.get_dataframes()
        return dataset[dataset_name]

    def clean_verses(self):
        """
        Remove all the bible references and the HTML tags in the verses
        """
        print("Cleaning...")
        p = Pool(3)
        self.dataframes = list(p.map(clean_df, self.dataframes))
        self._save_dfs()

    def align_versions(self):
        """
         The function align different version of the bible in a unique data frame
        """

        self.logs["align_versions"] = 'REFERENCE' + '-' * 15

        for i, _ in enumerate(self.dataframes):
            version = self.version_list[i].split('.')[0]
            self.logs["align_versions"] += ('\t' + version + '\t')

        self.logs["align_versions"] += '\n\n'

        start = time.time()
        try:
            ref_df = self.dataframes[0][1]
        except IndexError:
            raise ValueError("Check your director it might be empty.")
        books = set(ref_df['Book'].to_numpy().astype(int))

        for book in books:
            chapters = set(ref_df[(ref_df['Book'] == book)]['Chapter'].to_numpy().astype(int))

            for chapter in chapters:
                verses = ref_df[(ref_df['Book'] == book) &
                                (ref_df['Chapter'] == chapter)]['Verse'].to_numpy().astype(int)

                for verse in verses:
                    current_ref = '\r' + 'Book: {} Chapter: {} Verse: {}'.format(book, chapter, verse)

                    self.logs["align_versions"] += current_ref.ljust(20, '-')

                    sys.stdout.write(current_ref)
                    sys.stdout.flush()
                    self._group_ref((book, chapter, verse))

        spent_time = '\tTime spent: {:.2f}min'.format((time.time() - start) / 60)
        print(spent_time)

        self.logs["align_versions"] += spent_time

        self.aligned_df.to_csv('_'.join([column.split('.')[0] for column in self.aligned_df.columns[3:]]) +
                               '_' + 'aligned.csv', index=False, index_label=False,
                               encoding='utf8')

        return self.aligned_df

    def _group_ref(self, ref):
        """
        Function groups several verses from the bible to join it in single csv

        """

        book, chapter, verse = ref
        new_verses = {'Book': str(book),
                      'Chapter': str(chapter),
                      'Verse': str(verse)}

        for i, df in enumerate(self.dataframes):
            version, df = df

            try:

                scripture = df[(df['Book'] == book)
                               & (df['Chapter'] == chapter)
                               & (df['Verse'] == verse)]['Scripture'].to_numpy().tolist()[0]
                scripture = scripture.strip().replace('\"', '')
                new_verses[version.split('.')[0]] = scripture

            except IndexError:

                self.logs["align_versions"] += '-Missing in version: ' + version
                self.logs["align_versions"] += '\tFAIL'
                self.aligned_df = self.aligned_df[:-1]
                return False
                # print('-Missing in version: ' + version)

                # pdb.set_trace()

            except KeyError:
                self.logs["align_versions"] += '-Missing in version: ' + version
                self.logs["align_versions"] += '\tFAIL'
                self.aligned_df = self.aligned_df[:-1]
                return False
                # pdb.set_trace()

        self.aligned_df = self.aligned_df.append(new_verses, ignore_index=True)

        return True

    def save_report(self):
        """
         Saves a report of the process execution.
        """
        ctime = re.sub('[:]', '.', str(datetime.datetime.now()))
        try:
            os.mkdir('logs')
        except OSError:
            pass
        file = open(os.path.join('logs', 'report {}.txt'.format(ctime)), 'w')

        for keys in self.logs.keys():
            file.write(keys + '\n\n')
            file.write(''.ljust(12, '\t'))
            file.write('-' * 40)
            file.write('\n\n')
            file.write(self.logs[keys])
        file.close()

    @staticmethod
    def _replace_verse(ref, frame):
        """
        Finds the row with the reference and replaces the verse with the joined verse.
        Also, it deletes the old indexes where the joined verse was found.
        returns: new frame with the new verse.
        """
        frame_version, frame = frame
        frame_copy = frame.copy()

        ref_version, ref = ref
        book, chapter, verses = ref

        indexes = frame.index[(frame['Book'] == book) &
                              (frame['Chapter'] == chapter) &
                              (frame['Verse'] == list(verses)[0])].to_list()

        if frame_version != ref_version:
            texts = [frame[(frame['Book'] == book) &
                           (frame['Chapter'] == chapter) &
                           (frame['Verse'] == verse)]['Scripture'].to_numpy().tolist()
                     for verse in verses]

            if len(texts[-1]) == 0:
                raise Warning("Reference not found. Skipping..." + str(ref))

            new_verse_text = ' '.join(list(itertools.chain.from_iterable(texts)))

            if len(indexes) > 1:
                raise ValueError("More than one index found. {}".format(indexes))
            if len(indexes) == 0:
                raise Warning("Reference not found. Skipping...\n")

            frame_copy.loc[indexes, 'Scripture'] = new_verse_text

        n_repeated_verses = len(ref[-1])
        indexes_to_drop = list(range(indexes[0] + 1, indexes[0] + n_repeated_verses))
        frame = frame_copy.drop(indexes_to_drop)

        return frame_version, frame

    def to_join(self, ref):
        func_log = ""

        for i, frame in enumerate(self.dataframes):
            try:
                self.dataframes[i] = self._replace_verse(ref, frame)
            except Warning:
                func_log += "\nReference not found. Skipping..." + str(ref) + '-' + \
                            self.version_list[i] + '\n'
            except KeyError:
                func_log += '\nKeyError when dropping the indexes: ' + str(ref) + '\n'

            except IndexError:
                func_log += "\nNOT REPLACED AT {} VERSION {}".format(ref, self.version_list[i]) + '\n'
            except AssertionError:
                func_log += "\nNOT REPLACED AT {} VERSION {}".format(ref, self.version_list[i]) + '\n'
            except ValueError:
                print("Value Error")
        return func_log

    def join_verses(self):
        # function does not work properly
        """
         Function joins the verses cross all version of the bible to align them in a unique pattern when
         it is necessary.
         return: the list of dataframe preprocessed.
        """

        self.logs["_replace_verse"] = ""
        start = time.time()
        p = Pool(3)

        all_references = list(itertools.chain.from_iterable(p.map(find_joined_ref, self.dataframes)))

        for ref in set(all_references):
            self.logs["_replace_verse"] += self.to_join(ref)

        self._save_dfs()
        spent_time = (time.time() - start) / 60
        tot_spend = 'Total spent time: {:.2f}min'.format(spent_time)
        print(tot_spend)

        return self.dataframes

    def _save_dfs(self):
        print('Saving dataframes...')
        try:
            print('Creating dataset...')
            os.makedirs(self.output_dir)
            
        except OSError:
            pass
        for i, df in enumerate(self.dataframes):
            version_df, df = df
            df.to_csv(os.path.join(os.path.join(self.output_dir, version_df)), index_label=False, index=False)

    def to_txt_file(self, dir_path=os.curdir):
        """
            Convert the Scripture text to a .txt file
        """

        df = self.aligned_df
        for version in self.version_list:
            scripture = df[version].tolist()
            path = os.path.join(dir_path, version) + '.txt'

            from_text_to_file(scripture, path)

    def run_pipeline(self):

        self.join_verses()
        self.clean_verses()
        self.align_versions()

        self._save_dfs()
        self.save_report()
        return self.aligned_df
