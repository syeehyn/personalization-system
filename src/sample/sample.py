import pyspark.ml as M
import pyspark.sql.functions as F
import pyspark.sql.types as T

def sampling(ratings, min_user, min_movie, user_threshold, movie_threshold, random_seed):
    userid_filter = ratings.groupby('userId')\
        .agg(F.count(F.col('movieId'))\
                    .alias('count'))\
        .where(F.col('count') >= user_threshold)
    movieid_filter = ratings.groupby('movieId')\
            .agg(F.count(F.col('userId'))\
                        .alias('count'))\
            .where(F.col('count') >= movie_threshold)
    subset = ratings.join(userid_filter,
                            ratings.userId == userid_filter.userId)\
                        .select(ratings.userId, ratings.movieId, ratings.rating)\
                        .join(movieid_filter,
                            ratings.movieId == movieid_filter.movieId)\
                        .select(ratings.userId, ratings.movieId, ratings.rating).persist()
    frac = .001
    frac_incr = 2
    cnt_user = 0
    cnt_movie = 0
    while cnt_user < min_user or cnt_movie < min_movie:
        frac *= frac_incr
        sample = subset.sample(fraction=frac, seed = random_seed)
        userid_filter = sample.groupby('userId')\
                .agg(F.count(F.col('movieId'))\
                            .alias('count'))\
                .where(F.col('count') >= user_threshold)
        movieid_filter = sample.groupby('movieId')\
                .agg(F.count(F.col('userId'))\
                            .alias('count'))\
            .where(F.col('count') >= movie_threshold)
        sample = sample.join(userid_filter,
                            sample.userId == userid_filter.userId)\
                        .select(sample.userId, sample.movieId, sample.rating)\
                        .join(movieid_filter,
                            sample.movieId == movieid_filter.movieId)\
                        .select(sample.userId, sample.movieId, sample.rating)
        cnt_user = sample.select('userId').distinct().count()
        cnt_movie = sample.select('movieId').distinct().count()
        print(f'''
                with frac = {frac},
                {cnt_user} users rated at least {user_threshold} movies,
                {cnt_movie} movies are rated by at least {movie_threshold} users.
                ''')
    subset.unpersist()
    print(f'final sample has {cnt_user} users and {cnt_movie} movies')
    return sample