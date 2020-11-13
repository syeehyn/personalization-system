import json
import os
from os import path as osp
import sys
import pandas as pd
from src.utils import downloads, loading, Spark, train_test_split
from src.sample import sampling
from src.evaluation import Evaluator, Cross_validate_als

DOWNLOAD = json.load(open('config/downloads.json'))
SAMPLE = json.load(open('config/sample.json'))
SPLIT = json.load(open('config/split.json'))

def main(targets):
    """[main function to execute ETL pipeline]

    Args:
        targets (list): [list of string of commands]
    """
    if 'download' in targets:
        downloads(DOWNLOAD['url'], DOWNLOAD['fp'])
        if osp.isfile(osp.join(DOWNLOAD['fp'], 'movies.csv')):
            os.rename(osp.join(DOWNLOAD['fp'], 'movies.csv'), osp.join(SAMPLE['op'], 'movies.csv'))
    if 'sample' in targets:
        spark =Spark()
        ratings = loading(spark, DOWNLOAD['fp'])['ratings']
        sample = sampling(ratings,
                        SAMPLE['num_user'],
                        SAMPLE['num_movie'],
                        SAMPLE['user_threshold'],
                        SAMPLE['item_threshold'],
                        SAMPLE['random_seed'])
        sample = sample.toPandas()
        sample.to_csv(osp.join(SAMPLE['op'], 'sample.csv'), index = False)
    if 'train-test-split' in targets:
        spark = Spark()
        data = loading(spark, SAMPLE['op'])['sample']
        for i in SPLIT['splits']:
            train, test = train_test_split(data, i)
            train.toPandas().to_csv(osp.join(SPLIT['op'],
                                        'train_' + str(i) + '_' + str(1-i)+'.csv'), index=False)
            test.toPandas().to_csv(osp.join(SPLIT['op'],
                                        'test_' + str(i) + '_' + str(1-i)+'.csv'), index=False)
if __name__=='__main__':
    targets = sys.argv[1:]
    main(targets)
