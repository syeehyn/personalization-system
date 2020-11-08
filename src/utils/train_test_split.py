def train_test_split(data, seed, split):
    train = {}
    test = {}
    train, test = data.randomSplit([split, 1-split], seed=seed)
    return train, test