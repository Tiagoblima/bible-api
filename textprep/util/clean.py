import sys
from multiprocessing.dummy import Pool
from multiprocessing import Value


class CleanText:
    """This class is responsible for clean
    the whole text of a bible dataframe"""

    bible_df = None
    regex_set = None

    def __init__(self, dfs, regex_set):
        self.bible_df = dfs

        if regex_set is None:
            Value("There's no pattern to be removed.")

        self.regex_set = regex_set

    def _remove_regex(self, name_dataset):
        name = name_dataset[0]
        dataset = name_dataset[1]

        for exp in self.regex_set:
            dataset.replace(to_replace=exp, value=' ', regex=True, inplace=False)
        return name, dataset

    def clean_text(self):
        p = Pool(5)

        clean_dfs = p.map(self._remove_regex, self.bible_df.items())

        return dict(clean_dfs)
