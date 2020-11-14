import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS
import pyspark.sql.window as W

class Als():
    """[the predictor for Pyspark ALS]
    """
    def __init__(self, userCol, itemCol, ratingCol, regParam, seed, rank):
        self.userCol = userCol
        self.itemCol = itemCol
        self.ratingCol = ratingCol
        self.model =None
        self.als = ALS(userCol=userCol,
                itemCol=itemCol,
                ratingCol=ratingCol,
                coldStartStrategy="drop",
                nonnegative=True,
                regParam=regParam,
                seed=seed,
                rank=rank)
    def fit(self, _X):
        """[function to train parameter of predictor]

        Args:
            _X (Pyspark DataFrame): [training set]
        """
        self.X = _X
        X = self._preprocess(_X)
        self.model = self.als.fit(X)
    def predict(self, _X):
        """[function to make predict over test set]

        Args:
            _X (Pyspark DataFrame): [test set]

        Returns:
            Pyspark DataFrame: [DataFrame with 'prediction' column which has the predicting value]
        """        
        X = self._preprocess(_X)
        return self.model.transform(X)
    def recommend(self, k):
        pred = self.predict(self.X)
        window = W.Window.partitionBy(pred[self.userCol]).orderBy(pred['prediction'].desc())
        ranked = pred.select('*', F.rank().over(window).alias('rank'))
        recommended = ranked.where(ranked.rank <= k).select(F.col(self.userCol).cast('string'), 
                                                            F.col(self.itemCol).cast('string'))
        return recommended.groupby(self.userCol).agg(F.collect_list(self.itemCol).alias('recommendation'))

    def _preprocess(self, _X):
        """[preprocess the input dataset]

        Args:
            _X (Pyspark DataFrame): [the training or test set]

        Returns:
            Pyspark DataFrame: [the preprocessed DataFrame]
        """        
        cast_int = lambda df: df.select([F.col(c).cast('int') for c in [self.userCol, self.itemCol]] + \
                                [F.col(self.ratingCol).cast('float')])
        return cast_int(_X)
