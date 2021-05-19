import itertools
import os
import re

import pandas as pd

from textprep.util.util import preprocess_sentence


# -------------- TEXT FILE ----------------------


def join_verses(frame, ref):
    """
    Finds the row with the reference and replaces the verse with the joined verse.
    Also, it deletes the old indexes where the joined verse was found.
    returns: new frame with the new verse.
    """

    logs = 'LOGS util/replace_verse: '
    result = '\tSUCCESS\n'
    verses_texts = [frame[(frame['Book'] == ref[0][0]) &
                          (frame['Chapter'] == ref[0][1]) &
                          (frame['Verse'] == verse)]['Scripture'].to_numpy().tolist()
                    for verse in ref[-1]]

    success = True

    if len(verses_texts[-1]) == 0:
        logs += "Reference not found. Skipping..."
        return frame, logs + '\tFAIL'

    verses_texts = list(itertools.chain.from_iterable(verses_texts))

    new_verse_text = ' '.join(verses_texts)

    verse_list = list(ref[-1])

    indexes = frame.index[(frame['Book'] == ref[0][0]) &
                          (frame['Chapter'] == ref[0][1]) &
                          (frame['Verse'] == verse_list[0])].to_list()
    frame_copy = frame.copy()

    assert len(indexes) <= 1, "More than one index found. {}".format(indexes)

    if len(indexes) is 1:

        frame_copy.loc[indexes, 'Scripture'] = new_verse_text
        n_repeated_verses = len(ref[-1])
        indexes_to_drop = list(range(indexes[0] + 1, indexes[0] + n_repeated_verses))
        try:
            frame = frame_copy.drop(indexes_to_drop)
        except KeyError:
            success = False
            logs += 'KeyError when dropping the indexes: ' + str(indexes_to_drop)

    else:
        success = False
        logs += "Reference not found. Skipping...\n"

    if not success:
        result = '\tFAIL\n'

    return frame, logs + result


def find_joined_ref(dataframe):
    """
    Finds joined references using the pattern <sup>(.*)</sup>
    params:
    dataframe: a dataframe with the Following Header: Book, Chapter, Scripture
    return: list with tuple of range of verses that are joined.
    """

    searches = []

    for i, line in dataframe.iterrows():

        text = line['Scripture']

        found = re.findall(r'\[[0-9][0-9]*\-[0-9][0-9]*\]', text)

        for search in found:
            v_range = re.findall(r'[0-9][0-9]*', search)
            v_range[0] = int(v_range[0])
            v_range[1] = int(v_range[1])
            if v_range[0] > v_range[1]:
                continue

            book, chapter = line['Book'], line['Chapter']
            searches.append((book, chapter, range(v_range[0], v_range[1] + 1)))

    return searches


def get_references():
    """
         Remove the bible references cross the raw text.
        """

    books = []
    law = ['Gênesis', 'Êxodo', 'Levítico', 'Números', 'Deuteronômio']

    history = ['Josué', 'Juízes', 'Rute', '1 e 2 Samuel', '1 e 2 Reis', '1 e 2 Crônicas', 'Esdras', 'Neemias',
               'Tobias',
               'Judite', 'Ester', '1 e 2 Macabeus']

    poetry = ['Jó', 'Salmo', 'Provérbios', 'Eclesiastes', 'Cântico dos Cânticos', 'Sabedoria', 'Eclesiástico']

    prophets = ['Isaías', 'Jeremias', 'Lamentações', 'Baruc', 'Ezequiel', 'Daniel', 'Oséias', 'Joel', 'Amós',
                'Abdias',
                'Jonas', 'Miquéias', 'Naum', 'Habacuque', 'Sofonias', 'Ageu', 'Zacarias', 'Malaquias']

    gospel = ['Mateus', 'Marcos', 'Lucas', 'João']

    letters = ['Atos', 'Romanos', '1 e 2 Coríntios', 'Gálatas', 'Efésios', 'Filipenses', 'Colossenses',
               '1 e 2 Tessalonicenses', '1 e 2 Timóteo', 'Tito', 'Filemon', 'Hebreus', 'Tiago', '1 e 2 Pedro',
               '1 a 3 João', 'Judas', 'Apocalipse']

    books.extend(law)
    books.extend(history)
    books.extend(poetry)
    books.extend(prophets)
    books.extend(gospel)
    books.extend(letters)

    refs = []
    for book in books:
        book += ' [0-9]*[.]*[0-9]*[-]*[0-9]*[;]*'
        book = book.replace('1 e 2', '[0-9]')
        book = book.replace('1 a 3', '[0-9]')
        cleaner = re.compile(book)
        refs.append(cleaner)
    return refs


def remove_references(raw_text):
    """
     Remove the bible references cross the raw text.
    """

    for cleaner in get_references():
        raw_text = re.sub(cleaner, '', raw_text)

    return raw_text


def remove_tags(raw_text):
    cleaner = re.compile(r'<sup>\(.*\)</sup>')
    new_text = re.sub(cleaner, '', raw_text)

    cleaner = re.compile(r'<.*?>|\(-\)')

    return ' '.join(re.sub(cleaner, '', new_text).split())


path_to_pairs = '../Resources/pairs'
path_to_aligned = '../Resources/preprocessed/aligned_version.csv'
