from src.utils import downloads, loading, Spark
from src.sample import sampling
import json
import sys

DOWNLOAD = json.load(open('config/downloads.json'))
SAMPLE = json.load(open('config/sample.json'))



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
        sampling(ratings, SAMPLE['op'], SAMPLE['num_items'], SAMPLE['num_users'])

if __name__=='__main__':
    targets = sys.argv[1:]
    main(targets)