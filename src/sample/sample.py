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
            targetCol='rating'):
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
                            .select(ratings[userCol], ratings[itemCol], ratings[targetCol])
        userid_filter = sample.groupby(userCol)\
                        .agg(F.count(itemCol)\
                        .alias('cnt'))\
                        .where(F.col('cnt') >= user_threshold)\
                        .select(userCol)\
                        .orderBy(F.rand(seed=random_seed))\
                        .limit(num_user)
        sample = sample.join(userid_filter,
                                ratings[userCol] == userid_filter[userCol])\
                            .select(ratings[userCol], ratings[itemCol], ratings[targetCol]).persist()
        n_users, n_items = sample.select(userCol).distinct().count(), sample.select(itemCol).distinct().count()
        print(f'sample has {n_users} users and {n_items} items')
    return sample