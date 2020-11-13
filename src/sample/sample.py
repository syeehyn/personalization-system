import pyspark.ml as M
import pyspark.sql.functions as F
import pyspark.sql.types as T

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
    while n_users < num_user and n_items < num_item:
        movieid_filter = ratings.groupby(itemCol)\
            .agg(F.count(userCol)\
            .alias('cnt'))\
            .where(F.col('cnt') >= item_threshold)\
            .select(itemCol)\
            .orderBy(F.rand(seed=random_seed))\
            .limit(num_item)
        sample = ratings.join(movieid_filter,
                                ratings[itemCol] == movieid_filter[itemCol])\
                            .select(ratings[userCol], ratings[itemCol], ratings[timeCol], ratings[targetCol])
        userid_filter = sample.groupby(userCol)\
                        .agg(F.count(itemCol)\
                        .alias('cnt'))\
                        .where(F.col('cnt') >= user_threshold)\
                        .select(userCol)\
                        .orderBy(F.rand(seed=random_seed))\
                        .limit(num_user)
        sample = sample.join(userid_filter,
                                ratings[userCol] == userid_filter[userCol])\
                            .select(ratings[userCol], ratings[itemCol], ratings[timeCol], ratings[targetCol]).persist()
        n_users, n_items = sample.select(userCol).distinct().count(), sample.select(itemCol).distinct().count()
        print(f'sample has {n_users} users and {n_items} items')
    return sample