import pyspark.sql.functions as F
import pyspark.sql.window as W
def train_test_split(ratings, split, usercol='userId', itemcol='movieId', timecol='timestamp', targetcol='rating'):
    """[function to make train test split with respect to user activities]

    Args:
        ratings (Pyspark DataFrame): [the rating DataFrame to be splitted]
        split (float): [proportion of training set]
        usercol (str, optional): [user column name]. Defaults to 'userId'.
        itemcol (str, optional): [item column name]. Defaults to 'movieId'.
        timecol (str, optional): [timestamp column name]. Defaults to 'timestamp'.
        targetcol (str, optional): [rating/target column name]. Defaults to 'rating'.

    Returns:
        [Pyspark DataFrame, PysparkDataFrame]: [description]
    """    
    window = W.Window.partitionBy(ratings[usercol]).orderBy(ratings[timecol].desc())
    ranked = ratings.select('*', F.rank().over(window).alias('rank'))
    rating_count = ratings.groupby(usercol).agg(F.count(itemcol).alias('cnt'))
    ranked = ranked.join(rating_count, ranked.userId == rating_count.userId)\
        .select(ranked[usercol], ranked[itemcol], ranked[targetcol], ranked.rank, rating_count.cnt)
    ranked = ranked.withColumn('position', 1 - F.col('rank')/F.col('cnt'))\
        .select(usercol, itemcol,targetcol, 'position')
    train = ranked.where(ranked.position < split).select(usercol, itemcol, targetcol)
    test = ranked.where(ranked.position >= split).select(usercol, itemcol, targetcol)
    return train, test