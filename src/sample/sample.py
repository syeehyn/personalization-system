import pyspark.ml as M
import pyspark.sql.functions as F
import pyspark.sql.types as T

def sampling(ratings, op, num_items, num_users):
    rating_user_count = ratings.groupby('userId').agg(
            F.count(F.col('movieId')).alias('movie_count')
    )
    rating_item_count = ratings.groupby('movieId').agg(
                F.count(F.col('userId')).alias('user_count')
    )
    userid = rating_user_count.sort(F.col('movie_count').desc()).limit(num_users)
    itemid = rating_item_count.sort(F.col('user_count').desc()).limit(num_items)
    sample = ratings.join(userid, ratings.userId == userid.userId, 'inner')\
                .select(ratings.userId, ratings.movieId, ratings.rating)\
                .join(itemid, ratings.movieId == itemid.movieId, 'inner')\
                .select(ratings.userId, ratings.movieId, ratings.rating)
    stringIndexer = M.feature.StringIndexer(inputCol='userId', outputCol='userId_indx')
    model = stringIndexer.fit(sample)
    sample = model.transform(sample)
    stringIndexer = M.feature.StringIndexer(inputCol='movieId', outputCol='movieId_indx')
    model = stringIndexer.fit(sample)
    sample = model.transform(sample)
    sample = sample.select('userId', 'movieId', \
                            F.col('userId_indx').cast(T.IntegerType()),
                            F.col('movieId_indx').cast(T.IntegerType()),
                            'rating')\
                    .orderBy(['userId_indx', 'movieId_indx'])
    sample.coalesce(1).write.csv(path=op, compression='gzip', mode='overwrite', header = True)