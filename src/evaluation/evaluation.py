import pyspark.sql.functions as F
from tqdm import tqdm
import pandas as pd
from ..model_based import Als

def rmse(with_pred_df, rating_col_name = "rating", pred_col_name = "prediction"):
    """[calculate rmse of the prediction]

    Args:
        with_pred_df (Pyspark DataFrame): [Pyspark DataFrame with target and prediction columns]
        rating_col_name (str, optional): [column of true values]. Defaults to "rating".
        pred_col_name (str, optional): [column of prediction values]. Defaults to "prediction".

    Returns:
        flaot: [rmse]
    """
    return with_pred_df.select(F.sqrt(F.sum((F.col(rating_col_name) - \
                        F.col(pred_col_name))**2)/F.count(rating_col_name))).collect()[0][0]

def acc(with_pred_df, rating_col_name = "rating", pred_col_name = "prediction"):
    """[calculate rmse of the prediction]

    Args:
        with_pred_df (Pyspark DataFrame): [Pyspark DataFrame with target and prediction columns]
        rating_col_name (str, optional): [column of true values]. Defaults to "rating".
        pred_col_name (str, optional): [column of prediction values]. Defaults to "prediction".

    Returns:
        float: [accuracy]
    """
    TP = ((F.col(rating_col_name) >= 3) & (F.col(pred_col_name) >= 3))
    TN = ((F.col(rating_col_name) < 3) & (F.col(pred_col_name) < 3))
    correct = with_pred_df.filter(TP | TN)
    return correct.count() / with_pred_df.count()

class Evaluator():
    """[the evaluator for evaluation purpose]
    """    
    def __init__(self, metrics, ratingCol='rating', predCol='prediction'):
        self.metrics = metrics
        self.ratingCol = ratingCol
        self.predCol = predCol
    def evaluate(self, X):
        """[to evaluate the prediction]

        Args:
            X (Pyspark DataFrame): [Pyspark DataFrame with target and prediction columns]

        Raises:
            NotImplementedError: [metrics not yet implemented]

        Returns:
            float: [the corresponding metrics results]
        """        
        if self.metrics == 'rmse':
            return rmse(X, self.ratingCol, self.predCol)
        elif self.metrics =='accuracy':
            return acc(X, self.ratingCol, self.predCol)
        else:
            raise NotImplementedError

def Cross_validate_als(training, 
                        test, 
                        valid_ratio, 
                        regParam, 
                        rank, 
                        seed, 
                        evaluators,
                        userCol="userId",
                        itemCol="movieId",
                        ratingCol="rating"):
    """[helper function to tuning parameters of als algorithm]

    Args:
        training (Pyspark DataFrame): [the training set]
        test (Pyspark DataFrame): [the test set]
        valid_ratio (float): [ratio of validation set]
        regParam (list): [list of regParam parameters]
        rank (list): [list of rank parameters]
        seed (int): [random seed]
        evaluator (list): [list the evaluators]
        userCol (str, optional): [user column]. Defaults to "userId".
        itemCol (str, optional): [item column]. Defaults to "movieId".
        ratingCol (str, optional): [rating/target column]. Defaults to "rating".

    Returns:
        pd.DataFrame: [the pandas DataFrame with result of cross validation]
    """ 
    param_list = [(i, j) for i in regParam for j in rank]
    _training, _validation = training.randomSplit([1-valid_ratio, valid_ratio], seed = seed)
    result = []
    for i,j in tqdm(param_list):
        model = Als(userCol=userCol,
                itemCol=itemCol,
                ratingCol=ratingCol,
                regParam=i,
                seed=seed,
                rank=j
                )
        model.fit(_training)
        prediction = model.predict(_validation)
        res = []
        for k in evaluators:
            res.append(k.evaluate(prediction))
        result.append(res)
    return pd.DataFrame(result, index=param_list, columns = [e.metrics for e in evaluators])
