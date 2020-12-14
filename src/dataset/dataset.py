from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder
from scipy import sparse
import pandas as pd
import numpy as np
import torch

class MovielensDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, 
                        genre_path, 
                        train, 
                        genre_encoder=None, 
                        ordinal_encoder=None,
                        binary=False,
                        validation=0.1,
                        user_col = 'userId',
                        item_col = 'movieId',
                        rating_col = 'rating'):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.train = train
        self.binary = binary
        data = pd.read_csv(dataset_path)
        movies = pd.read_csv(genre_path)
        if genre_encoder:
            self.genre_encoder = genre_encoder
        else:
            self.genre_encoder = CountVectorizer()
        if ordinal_encoder:
            self.indexer = ordinal_encoder
        else:
            self.indexer = OrdinalEncoder()
        if binary:
            self.targets = self.__preprocess_target(data[rating_col].values).astype(np.float32)
        else:
            self.targets = data[rating_col].values.astype(np.float32)
        
        self.items = self.__preprocess_id(data[[user_col, item_col]])
        self.genres = self.__preprocess_genre(data, movies)
        self.field_dims = np.max(self.items, axis=0) + 1
        if validation:
            self.val_mask = np.unique(np.random.choice(range(len(self)), int(len(self)*.01), replace=False))
            self.train_mask = list(set(range(len(self))) - set(self.val_mask))
        else:
            self.val_mask = []
            self.train_mask = list(range(len(self)))
        
    def __len__(self):
        return self.targets.shape[0]
    def __getitem__(self, index):
        return self.items[index], self.genres[index].toarray(), self.targets[index]
    
    def __preprocess_genre(self, X, movies):
        if self.train:
            onehot_genre = self.genre_encoder.fit_transform(movies.genres).toarray()
        else:
            onehot_genre = self.genre_encoder.transform(movies.genres).toarray()
        onehot_genre = pd.DataFrame(onehot_genre, columns = [f'genre_{i}' for i in range(onehot_genre.shape[1])])
        onehot_genre[self.item_col] = movies[self.item_col]
        merged = X.merge(onehot_genre, how = 'left')
        genre_col = [col for col in merged if col.startswith('genre')]
        onehot_genre = merged[genre_col].values
        onehot_genre = sparse.csr_matrix(onehot_genre, onehot_genre.shape)
        return onehot_genre
    def __preprocess_id(self, X):
        if self.train:
            return self.indexer.fit_transform(X).astype(np.long)
        else:
            output = np.zeros(X.shape)
            known = X[X[self.user_col].isin(self.indexer.categories_[0]) & \
                    X[self.item_col].isin(self.indexer.categories_[1])].index.tolist()
            unknown_user = X[~X[self.user_col].isin(self.indexer.categories_[0])].index.tolist()
            unknown_movie = X[~X[self.item_col].isin(self.indexer.categories_[1])].index.tolist()
            output[known, :] = self.indexer.transform(X.iloc[known])
            output[unknown_user, 0] = -1
            output[unknown_user, 1] = X.iloc[unknown_user][self.item_col]
            output[unknown_movie, 0] = X.iloc[unknown_movie][self.user_col]
            output[unknown_movie, 1] = -1
            return output.astype(np.long)
    def __preprocess_target(self, target):
        target[target < 3] = 0
        target[target >= 3] = 1
        return target