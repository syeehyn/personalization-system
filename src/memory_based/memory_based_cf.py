from scipy import sparse
import numpy as np
from tqdm import tqdm
import pyspark.sql.functions as F
import warnings
from .. import indexTransformer
warnings.filterwarnings("ignore")
class Memory_based_CF():
    def __init__(self, spark, base, usercol='userId', itemcol='movieId', ratingcol='rating'):
        self.base = base
        self.usercol = usercol
        self.itemcol = itemcol
        self.ratingcol = ratingcol
        self.spark = spark
        self.X = None
        self.idxer = None
        self.similarity_matrix = None
    def fit(self, _X):
        X = self._preprocess(_X, True)
        self.X = X
        self.similarity_matrix = self._pearson_corr(X)
    def predict(self, _X):
        rows, cols = self._preprocess(_X, False)
        mu = np.array(np.nan_to_num(self.X.sum(1) / (self.X != 0).sum(1))).reshape(-1)
        preds = []
        for i,j in tqdm(zip(rows,cols), total=len(rows)):
            row = sparse.find(self.X[:, j])[0]
            target = sparse.find(self.X[:, j])[2]
            nom = np.array(self.similarity_matrix[i, row]).reshape(-1).dot((target - mu[row]))
            denom = np.linalg.norm(self.similarity_matrix[i, row], ord = 1)
            if denom == 0:
                val = mu[i]
            else:
                val = mu[i] + nom/denom          
            preds.append(val)
        df = self.idxer.transform(_X).select(self.usercol, self.itemcol, self.ratingcol).toPandas()
        df['prediction'] = preds
        return self.spark.createDataFrame(df)
    def _preprocess(self, X, fit):
        cast_int = lambda df: df.select([F.col(c).cast('int') for c in [self.usercol, self.itemcol]] + \
                                [F.col(self.ratingcol).cast('float')])
        _X = cast_int(X)
        if fit:
            self.idxer = indexTransformer(self.usercol, self.itemcol)
            self.idxer.fit(_X)
            X = self.idxer.transform(_X).select(self.usercol+'_idx', self.itemcol+'_idx', self.ratingcol).toPandas().values
            if self.base == 'user':
                row = X[:, 0]
                col = X[:, 1]
                data = X[:, 2]
            elif self.base == 'item':
                row = X[:, 1]
                col = X[:, 0]
                data = X[:, 2]
            else:
                raise NotImplementedError
            return sparse.csr_matrix((data, (row, col)))
        else:
            X = self.idxer.transform(_X).select(self.usercol+'_idx', self.itemcol+'_idx').toPandas().values
            if self.base == 'user':
                row = X[:, 0]
                col = X[:, 1]
            elif self.base == 'item':
                row = X[:, 1]
                col = X[:, 0]
            else:
                raise NotImplementedError
            return row, col

    def _pearson_corr(self, A):
        n = A.shape[1]
        
        rowsum = A.sum(1)
        centering = rowsum.dot(rowsum.T) / n
        C = (A.dot(A.T) - centering) / (n - 1)
        
        d = np.diag(C)
        coeffs = C / np.sqrt(np.outer(d, d))
        return np.array(np.nan_to_num(coeffs))