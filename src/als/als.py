import pyspark.ml as M
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from tqdm import tqdm

cast_int = lambda df: df.select([F.col(c).cast('int') for c in ['userId', 'movieId']] + \
                                [F.col('rating').cast('float')])

def cross_validate_als(training_set, test_set, valid_ratio, regParam, rank, seed):
    training, test = cast_int(training_set), cast_int(test_set)
    print(f'''
        training set num of rows {training.count()},
        test set num of rows {test.count()},
        training set num of users {training.select('userId').distinct().count()},
        training set num of movies {training.select('movieId').distinct().count()},
        test set num of users {test.select('userId').distinct().count()},
        test set num of movies {test.select('movieId').distinct().count()},
        ''')
    param_list = [(i, j) for i in regParam for j in rank]
    _training, _validation = training.randomSplit([1-valid_ratio, valid_ratio], seed = seed)
    evaluator=RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    rmses = []
    for i,j in tqdm(param_list):
        als = ALS(userCol="userId", 
                itemCol="movieId", 
                ratingCol="rating",
                coldStartStrategy="drop", 
                nonnegative=True,
                regParam=i,
                seed=seed,
                rank=j)
        model = als.fit(_training)
        rmses.append(evaluator.evaluate(model.transform(_validation)))
    
    return dict(zip(param_list, rmses))