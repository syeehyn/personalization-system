import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS

class Als():
    """[summary]
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
        """[summary]

        Args:
            _X ([type]): [description]
        """        
        X = self._preprocess(_X)
        self.model = self.als.fit(X)
    def predict(self, _X):
        """[summary]

        Args:
            _X ([type]): [description]

        Returns:
            [type]: [description]
        """        
        X = self._preprocess(_X)
        return self.model.transform(X)
    def _preprocess(self, _X):
        """[summary]

        Args:
            _X ([type]): [description]

        Returns:
            [type]: [description]
        """        
        cast_int = lambda df: df.select([F.col(c).cast('int') for c in [self.userCol, self.itemCol]] + \
                                [F.col(self.ratingCol).cast('float')])
        return cast_int(_X)
