import json
import requests, zipfile, io
from tqdm import tqdm
from os import path as osp
from glob import glob
import sys
import pyspark.ml as M
import pyspark.sql.functions as F
import pyspark.sql.types as T
from spark_session import spark_session
DOWNLOAD = json.load(open('config/downloads.json'))
SAMPLE = json.load(open('config/sample.json'))
def sampling(ratings,
            num_user, 
            num_item, 
            user_threshold, 
            item_threshold, 
            random_seed,
            userCol='userId', 
            itemCol='movieId',
            timeCol = 'timestamp',
            targetCol='rating'):
    """[method to generating sample from BIG dataset]

    Args:
        ratings (Pyspark DataFrame): [the BIG dataset]
        num_user (int): [the number of users needs to have in the sample]
        num_item (int): [the number of items needs to have in the sample]
        user_threshold (int): [the number of ratings a user needs to have]
        item_threshold (int): [the number of ratings a movie needs to have]
        random_seed (int): [random seed of random sample]
        userCol (str, optional): [user column name]. Defaults to 'userId'.
        itemCol (str, optional): [item column name]. Defaults to 'movieId'.
        timeCol (str, optional): [timesampe column name]. Defaults to 'timestamp'.
        targetCol (str, optional): [rating/target column name]. Defaults to 'rating'.

    Returns:
        Pyspark DataFrame: [the sample]
    """    
    n_users, n_items = 0, 0
    M, N = num_item, num_user
    while n_users < num_user and n_items < num_item:
        movieid_filter = ratings.groupby(itemCol)\
            .agg(F.count(userCol)\
            .alias('cnt'))\
            .where(F.col('cnt') >= item_threshold)\
            .select(itemCol)\
            .orderBy(F.rand(seed=random_seed))\
            .limit(M)
        sample = ratings.join(movieid_filter,
                                ratings[itemCol] == movieid_filter[itemCol])\
                            .select(ratings[userCol], ratings[itemCol], ratings[timeCol], ratings[targetCol])
        userid_filter = sample.groupby(userCol)\
                        .agg(F.count(itemCol)\
                        .alias('cnt'))\
                        .where(F.col('cnt') >= user_threshold)\
                        .select(userCol)\
                        .orderBy(F.rand(seed=random_seed))\
                        .limit(N)
        sample = sample.join(userid_filter,
                                ratings[userCol] == userid_filter[userCol])\
                            .select(ratings[userCol], ratings[itemCol], ratings[timeCol], ratings[targetCol]).persist()
        n_users, n_items = sample.select(userCol).distinct().count(), sample.select(itemCol).distinct().count()
        print(f'sample has {n_users} users and {n_items} items')
        M += 100
        N += 100
    return sample

def downloading(url, fp):
    """[to download zipfile with url]

    Args:
        url (str): [the url of the downloading file]
        fp (str): [the filepath of downloaded file]
    """    
    r = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zip:
        for zip_info in tqdm(zip.infolist()):
            if zip_info.filename[-1] == '/':
                continue
            zip_info.filename = osp.basename(zip_info.filename)
            zip.extract(zip_info, fp)
def loading(spark, fp):
    """[function to load data from a directory]

    Args:
        spark (Spark Session): [the current spark session]
        fp (str): [the file path of directory]

    Returns:
        dict: [the dictionary of loaded files]
    """    
    data_list = [i.split('/')[-1] for i in glob(osp.join(fp, '*.csv'))]
    data_map = {
    file_name[:-4]:spark.read.format('csv').option('header', 'true').load(osp.join(fp, file_name))\
    for file_name in data_list
    }
    return data_map

def main(targets):
    """[main function to execute ETL pipeline]

    Args:
        targets (list): [list of string of commands]
    """
    if 'download' in targets:
        downloading(DOWNLOAD['url'], DOWNLOAD['fp'])
    if 'sample' in targets:
        spark =spark_session()
        ratings = loading(spark, DOWNLOAD['fp'])['ratings']
        sample = sampling(ratings,
                        SAMPLE['num_user'],
                        SAMPLE['num_movie'],
                        SAMPLE['user_threshold'],
                        SAMPLE['item_threshold'],
                        SAMPLE['random_seed'])
        sample = sample.toPandas()
        sample.to_csv(osp.join(SAMPLE['op'], 'sample.csv'), index = False)

    
if __name__=='__main__':
    targets = sys.argv[1:]
    main(targets)
