def train_test_split(data, seed):
    train = {}
    test = {}
    for i in [.25, .5, .75]:
        train[i], test[1-i] = data.randomSplit([i, 1-i], seed=seed)
    return train, test