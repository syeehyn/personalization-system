from scipy import sparse
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
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
        self.prediction_matrix = self._get_predict()
        
    def predict(self, _X):
        rows, cols = self._preprocess(_X, False)
        preds = []
        for i,j in zip(rows,cols):   
            preds.append(self.prediction_matrix[i, j])
        df = self.idxer.transform(_X).select(self.usercol, self.itemcol, self.ratingcol).toPandas()
        df['prediction'] = preds
        return self.spark.createDataFrame(df)
    def _preprocess(self, X, fit):
        if fit:
            self.idxer = indexTransformer(self.usercol, self.itemcol)
            self.idxer.fit(X)
            _X = self.idxer.transform(X)\
                            .select(F.col(self.usercol+'_idx').alias(self.usercol), 
                                    F.col(self.itemcol+'_idx').alias(self.itemcol), 
                                    F.col(self.ratingcol))
            _X = _X.toPandas().values
            if self.base == 'user':
                row = _X[:, 0].astype(int)
                col = _X[:, 1].astype(int)
                data = _X[:, 2].astype(float)
            elif self.base == 'item':
                row = _X[:, 1].astype(int)
                col = _X[:, 0].astype(int)
                data = _X[:, 2].astype(float)
            else:
                raise NotImplementedError
            return sparse.csr_matrix((data, (row, col)))
        else:
            _X = self.idxer.transform(X).select(self.usercol+'_idx', self.itemcol+'_idx').toPandas().values
            if self.base == 'user':
                row = _X[:, 0].astype(int)
                col = _X[:, 1].astype(int)
            elif self.base == 'item':
                row = _X[:, 1].astype(int)
                col = _X[:, 0].astype(int)
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
        return np.array(np.nan_to_num(coeffs)) - np.eye(A.shape[0])
    def _get_predict(self):
        mu_iarray = np.array(np.nan_to_num(self.X.sum(1) / (self.X != 0).sum(1))).reshape(-1)
        mu_imat = np.vstack([mu_iarray for _ in range(self.X.shape[1])]).T
        x = self.X.copy()
        x[x==0] = np.NaN
        diff = np.nan_to_num(x-mu_imat)
        sim_norm_mat = abs(self.similarity_matrix).dot((diff!=0).astype(int))
        w = self.similarity_matrix.dot(diff) / sim_norm_mat
        w = np.nan_to_num(w)
        return mu_imat + w