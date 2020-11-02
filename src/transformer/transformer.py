import pyspark.ml as M
import pyspark.sql.functions as F
class indexTransformer():
    """[summary]
    """    
    def __init__(self, usercol='userId', itemcol='movieId'):
        """[summary]

        Args:
            usercol (str, optional): [description]. Defaults to 'userId'.
            itemcol (str, optional): [description]. Defaults to 'movieId'.
        """        
        self.usercol = usercol
        self.itemcol = itemcol
        self.u_indxer =  M.feature.StringIndexer(inputCol=usercol, 
                                                outputCol=usercol+'_idx', 
                                                handleInvalid = 'skip')
        self.i_indxer = M.feature.StringIndexer(inputCol=itemcol, 
                                                outputCol=itemcol+'_idx', 
                                                handleInvalid = 'skip')
        self.X = None
    def fit(self, X):
        """[summary]

        Args:
            X ([type]): [description]
        """        
        self.X = X
        self.u_indxer = self.u_indxer.fit(self.X)
        self.i_indxer = self.i_indxer.fit(self.X)
        return
    def transform(self, X):
        """[summary]

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """        
        X_ = self.u_indxer.transform(X)
        X_ = self.i_indxer.transform(X_)
        return self._cast_int(X_).orderBy([self.usercol+'_idx', self.itemcol+'_idx'])
    
    def fit_transform(self, X):
        """[summary]

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """        
        self.fit(X)
        return self.transform(X)
    
    def _cast_int(self, X):
        """[summary]

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return X.select([F.col(c).cast('int') for c in X.columns])