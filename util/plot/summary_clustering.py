from pathlib import Path
import re
import argparse
from glob import iglob
import pandas as pd
import numpy as np

def parse_args():
    '''
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='figures')
    parser.add_argument('--exclude', type=str, nargs='+', default=[])
    args = parser.parse_args()

    return args

def load_summaries(root_dir, exclude=[]):
    summaries = []
    for filename in map(Path, iglob(f'{root_dir}/*sim_*.csv')):
        if re.match(r'.*_\d+$', filename.stem):
            nvir = filename.stem.split('_')[-1]
            if nvir in exclude:
                continue
            summary = pd.read_csv(filename)
            summary['nb_viruses'] = nvir
            summaries.append(summary)
    return pd.concat(summaries)

def main():
    '''
    '''

    args = parse_args()
    data = load_summaries(args.dir, exclude=args.exclude)
    output = f'{args.dir}/summary_clustering_sim'

    if args.exclude:
        output += '_exclude-{}'.format('-'.join(args.exclude))

    (data
     .groupby(['metric', 'method'])
     .score
     .agg(min=np.min, max=np.max, mean=np.mean, median=np.median, IQR=lambda x: x.quantile(0.75)-x.quantile(0.25))
     .to_csv(f'{output}.csv'))


if __name__ == '__main__':
    main()
