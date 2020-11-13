import pyspark.ml as M
import pyspark.sql.functions as F
class indexTransformer():
    """[summary]
    """    
    def __init__(self, usercol='userId', itemcol='movieId', ratingcol='rating'):
        """[the index transformer for matrix purpose]

        Args:
            usercol (str, optional): [user column name]. Defaults to 'userId'.
            itemcol (str, optional): [item column name]. Defaults to 'movieId'.
        """        
        self.usercol = usercol
        self.itemcol = itemcol
        self.ratingcol = ratingcol
        self.u_indxer =  M.feature.StringIndexer(inputCol=usercol, 
                                                outputCol=usercol+'_idx', 
                                                handleInvalid = 'skip')
        self.i_indxer = M.feature.StringIndexer(inputCol=itemcol, 
                                                outputCol=itemcol+'_idx', 
                                                handleInvalid = 'skip')
        self.X = None
    def fit(self, X):
        """[to train the transformer]

        Args:
            X (Pyspark DataFrame): [the DataFrame for training]
        """        
        self.X = X
        self.u_indxer = self.u_indxer.fit(self.X)
        self.i_indxer = self.i_indxer.fit(self.X)
        return
    def transform(self, X):
        """[to transform the DataFrame]

        Args:
            X (Pyspark DataFrame): [the DataFrame needs to be transformed]

        Returns:
            Pyspark DataFrame: [the transformed DataFrame with index]
        """        
        X_ = self.u_indxer.transform(X)
        X_ = self.i_indxer.transform(X_)
        return X_.orderBy([self.usercol+'_idx', self.itemcol+'_idx'])
    
    def fit_transform(self, X):
        """[combining fit and transform]

        Args:
            X (Pyspark DataFrame): [the DataFrame needs to be trained and transformed]

        Returns:
            Pyspark DataFrame: [the transformed DataFrame with index]
        """        
        self.fit(X)
        return self.transform(X)