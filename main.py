import json
from os import path as osp
import sys
import pandas as pd
from src.utils import downloads, loading, Spark, train_test_split
from src.sample import sampling
from src.evaluation import Evaluator, Cross_validate_als

DOWNLOAD = json.load(open('config/downloads.json'))
SAMPLE = json.load(open('config/sample.json'))
SPLIT = json.load(open('config/split.json'))
ALS = json.load(open('config/als_params.json'))

def main(targets):
    """[summary]

    Args:
        targets ([type]): [description]
    """
    if 'download' in targets:
        downloads(DOWNLOAD['url'], DOWNLOAD['fp'])
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
            train, test = train_test_split(data, SPLIT['seed'], i)
            train.toPandas().to_csv(osp.join(SPLIT['op'],
                                        'train_' + str(i) + '_' + str(1-i)+'.csv'), index=False)
            test.toPandas().to_csv(osp.join(SPLIT['op'],
                                        'test_' + str(i) + '_' + str(1-i)+'.csv'), index=False)
    if 'cv-als' in targets:
        spark = Spark()
        datas = loading(spark, ALS['data_fp'])
        for i in ['0.75_0.25', '0.5_0.5', '0.25_0.75']:
            print(f'generating cv result for traing_test: {i}')
            train, test = datas['train_' + i], datas['test_' + i]
            evaluators = [Evaluator('rmse'), Evaluator('accuracy')]
            result = Cross_validate_als(train,
                                                test,
                                                ALS['valid_ratio'],
                                                ALS['regParam'],
                                                ALS['rank'],
                                                ALS['seed'],
                                                evaluators
                                                )
            result.to_csv(osp.join(ALS['op'], 'als_' + i + '_cv_result.csv'))
if __name__=='__main__':
    targets = sys.argv[1:]
    main(targets)
