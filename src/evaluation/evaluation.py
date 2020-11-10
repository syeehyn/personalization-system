import pyspark.sql.functions as F
from tqdm import tqdm
import pandas as pd
from ..model_based import Als
from ..model_based import nmf

def rmse(with_pred_df, rating_col_name = "rating", pred_col_name = "prediction"):
    """[summary]

    Args:
        with_pred_df ([type]): [description]
        rating_col_name (str, optional): [description]. Defaults to "rating".
        pred_col_name (str, optional): [description]. Defaults to "prediction".

    Returns:
        [type]: [description]
    """
    return with_pred_df.select(F.sqrt(F.sum((F.col(rating_col_name) - \
                        F.col(pred_col_name))**2)/F.count(rating_col_name))).collect()[0][0]

def acc(with_pred_df, rating_col_name = "rating", pred_col_name = "prediction"):
    """[summary]

    Args:
        with_pred_df ([type]): [description]
        rating_col_name (str, optional): [description]. Defaults to "rating".
        pred_col_name (str, optional): [description]. Defaults to "prediction".

    Returns:
        [type]: [description]
    """
    TP = ((F.col(rating_col_name) >= 3.5) & (F.col(pred_col_name) >= 3.5))
    TN = ((F.col(rating_col_name) < 3.5) & (F.col(pred_col_name) < 3.5))
    correct = with_pred_df.filter(TP | TN)
    return correct.count() / with_pred_df.count()

def coverage_k(with_pred_df, id_col_name, rating_col_name = "rating", 
                pred_col_name = "prediction", k=2):
    """[summary]

    Args:
        with_pred_df ([type]): [description]
        id_col_name ([type]): [description]
        rating_col_name (str, optional): [description]. Defaults to "rating".
        pred_col_name (str, optional): [description]. Defaults to "prediction".
        k (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    TP = ((F.col(rating_col_name) >= 3.5) & (F.col(pred_col_name) >= 3.5))
    num_covered = with_pred_df.select(id_col_name, rating_col_name, pred_col_name).filter(TP).groupBy(id_col_name).count()
    num_covered_bigger_than_k = num_covered.filter(f"count >= {k}")
    return num_covered_bigger_than_k.count() / num_covered.count()

class Evaluator():
    """[summary]
    """    
    def __init__(self, metrics, ratingCol='rating', predCol='prediction', idCol=None, k=None):
        self.metrics = metrics
        self.ratingCol = ratingCol
        self.predCol = predCol
        self.idCol = idCol
        self.k = k
    def evaluate(self, X):
        """[summary]

        Args:
            X ([type]): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """        
        if self.metrics == 'rmse':
            return rmse(X, self.ratingCol, self.predCol)
        elif self.metrics =='accuracy':
            return acc(X, self.ratingCol, self.predCol)
        elif self.metrics =='converage_k':
            return coverage_k(X, self.idCol, self.ratingCol, self.predCol, self.k)
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
    """[summary]

    Args:
        training ([type]): [description]
        test ([type]): [description]
        valid_ratio ([type]): [description]
        regParam ([type]): [description]
        rank ([type]): [description]
        seed ([type]): [description]
        evaluator ([type]): [description]
        userCol (str, optional): [description]. Defaults to "userId".
        itemCol (str, optional): [description]. Defaults to "movieId".
        ratingCol (str, optional): [description]. Defaults to "rating".

    Returns:
        [type]: [description]
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
