import pyspark.sql.functions as F


class Baseline():
    """[baseline model]
    """    
    def __init__(self, usercol='userId', itemcol='movieId', ratingcol='rating'):
        self.usercol = usercol
        self.itemcol = itemcol
        self.ratingcol = ratingcol
    
    def fit(self, X):
        """[train the model]

        Args:
            X (Pyspark DataFrame): [training set]
        """        
        train = self._preprocess(X)
        umean = train.groupby(self.usercol).agg(F.mean(self.ratingcol).alias('umean'))
        imean = train.groupby(self.itemcol).agg(F.mean(self.ratingcol).alias('imean'))
        self.umean = umean
        self.imean = imean
        
    def predict(self, X):
        """[predict based on X]

        Args:
            X (Pyspark DataFrame): [dataset to be predicted]

        Returns:
            [Pyspark DataFrame]: [predicted DataFrame with prediction column]
        """        
        test = self._preprocess(X)
        
        pred = test.join(self.umean, test[self.usercol] == self.umean[self.usercol])\
                        .select(test[self.usercol], test[self.itemcol], test[self.ratingcol], self.umean.umean)
        pred = pred.join(self.imean, pred[self.itemcol] == self.imean[self.itemcol])\
                        .select(pred[self.usercol], pred[self.itemcol], pred[self.ratingcol], pred.umean, self.imean.imean)
        
        pred = pred.select(pred[self.usercol], pred[self.itemcol], pred[self.ratingcol],
                        ((F.col('umean') + F.col('imean'))/2).alias('prediction'))
        
        return pred
        
        
        
    def _preprocess(self, _X):
        """[preprocess the input dataset]

        Args:
            _X (Pyspark DataFrame): [the training or test set]

        Returns:
            Pyspark DataFrame: [the preprocessed DataFrame]
        """        
        cast_int = lambda df: df.select([F.col(c).cast('int') for c in [self.usercol, self.itemcol]] + \
                                [F.col(self.ratingcol).cast('float')])
        return cast_int(_X)