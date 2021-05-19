import argparse
import getopt
import sys

import pandas as pd

from textprep.bible.preprocess import TextPreprocess

pd.set_option('display.max_colwidth', None)

pt_versions = versions = ['ntlh', 'nvi', 'aa', 'acf']


def run_preprocessing(base_dir, output_dir=None):
    if output_dir is None:
        output_dir = base_dir

    text_prep = TextPreprocess(base_dir,
                               output_dir=output_dir,
                               from_sqlite=False,
                               clean_out_dir=False,
                               load_files=True)

    return text_prep.run_pipeline()


BASE_DIR = '../Resources/texts'


def main(argv):
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--input_dir', type=str,
                        help='O ano de backup', required=True)
    parser.add_argument('--output_dir', type=str,
                        help='O ano de backup', required=True)
    parser.add_argument('--from_sqlite', action="store_true",
                        help='Não reseta o banco antes da modificação')
    args = parser.parse_args()

    print('Input dir is ', args.input_dir)
    print('Output dir is ', args.output_dir)

    aligned_df = run_preprocessing(args.input_dir, output_dir=args.output_dir)
    aligned_df.to_csv(args.output_dir + '_' + 'aligned_text.csv', index=False, index_label=False,
                      encoding='utf8')


if __name__ == "__main__":
    main(sys.argv[1:])
