import argparse
import os

import pandas as pd
import numpy as np

RANKS = ['phylum', 'class', 'order','family', 'genus']
DB_PATH = os.path.expanduser("~/db")

def parse_args():
    '''
    '''

    parser = argparse.ArgumentParser()
    # parser.add_argument('', type=int, default=0)
    args = parser.parse_args()

    return args

# def fill_na_ranks(row, ranks, values):
#     if not ranks:
#         return values
#     rank = ranks.pop()
#     if pd.isnull(row[rank]):
#         return fill_na_ranks(row, ranks, values)

def preprocess_lineages():
    data = (pd.read_csv(f"{DB_PATH}/lineages.csv", index_col=0)
            .drop('species', axis=1)
            .dropna(how='all')
            .fillna('N/A'))
    species_per_genus = data.apply('|'.join, axis=1)
    counts = species_per_genus.value_counts()
    data.index = counts.loc[species_per_genus].values
    data.to_csv(f'{DB_PATH}/lineages_cleaned.tsv', sep='\t')
    

def main():
    '''
    '''

    args = parse_args()
    preprocess_lineages()

if __name__ == '__main__':
    main()
