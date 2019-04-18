# coding: utf-8

import numpy as np
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge

from utils import *


def user_based(train):
    predicts = np.zeros(shape=train.shape)
    user_means = np.zeros(shape=(train.shape[0],))
    for i in range(user_means.shape[0]):
        user_means[i] = np.mean(train[i, train[i] > 0])

    for i in tqdm.tqdm(range(train.shape[0])):
        for j in range(train.shape[1]):
            if train[i, j] > 0:
                predicts[i, j] = train[i, j]
                continue

            predicts[i, j] = user_means[i]
            mask = train[:, j] > 0
            if sum(mask) == 0:
                continue

            sim = cosine_similarity(train[i].reshape(1, -1), train[mask])
            numerator = np.sum((train[mask, j] - user_means[mask]) * sim.reshape(-1, ))
            denominator = np.sum(sim)

            if denominator > 1e-6:
                predicts[i, j] += numerator / denominator
    return predicts


def item_based(train):
    predicts = np.zeros(shape=train.shape)
    item_means = np.zeros(shape=(train.shape[1],))
    for i in range(item_means.shape[0]):
        slic = train[train[:, i] > 0, i]
        if slic.size > 0:
            item_means[i] = np.mean(slic)

    for i in tqdm.tqdm(range(train.shape[0])):
        for j in range(train.shape[1]):
            if train[i, j] > 0:
                predicts[i, j] = train[i, j]
                continue

            predicts[i, j] = item_means[j]
            mask = train[i] > 0
            if sum(mask) == 0:
                continue

            sim = cosine_similarity(train[:, j].reshape(1, -1), train[:, mask].transpose())
            numerator = np.sum((train[i, mask] - item_means[mask]) * sim.reshape(-1, ))
            denominator = np.sum(sim)

            if denominator > 1e-6:
                predicts[i, j] += numerator / denominator
    return predicts


def linear(train):
    samples, y = [], []
    for i in range(train.shape[0]):
        for j in range(train.shape[1]):
            if train[i, j] > 0:
                sample = np.zeros(shape=(train.shape[0] + train.shape[1],))
                sample[i] = 1
                sample[train.shape[0] + j] = 1

                samples.append(sample)
                y.append(train[i, j])

    train_x, train_y = np.array(samples), np.array(y)
    model = Ridge(alpha=3)
    model.fit(train_x, train_y)

    results = train.astype(np.float)
    for i in tqdm.tqdm(range(train.shape[0])):
        for j in range(train.shape[1]):
            if train[i, j] == 0:
                sample = np.zeros(shape=(train.shape[0] + train.shape[1],))
                sample[i] = 1
                sample[train.shape[0] + j] = 1

                results[i, j] = model.predict(sample.reshape(1, -1))[0]
            else:
                results[i, j] = train[i, j]

    return results


def ials(train, test, init, n_iter=10, n_features=5, lambda_val=8, alpha=25):
    user_size, item_size = train.shape

    users = np.random.normal(size=(user_size, n_features))
    items = np.random.normal(size=(item_size, n_features))

    lambda_I = lambda_val * np.eye(n_features)
    confidence = np.ones(train.shape) + train * alpha

    for it in range(n_iter):
        for u in range(user_size):
            A = np.dot(items.T * confidence[u], items) + lambda_I * (train[u] > 0).sum()
            b = np.dot(items.T * confidence[u], init[u])
            users[u] = np.linalg.solve(A, b)

        for i in range(item_size):
            A = np.dot(users.T * confidence[:, i], users) + lambda_I * (train[:, i] > 0).sum()
            b = np.dot(users.T * confidence[:, i], init[:, i])
            items[i] = np.linalg.solve(A, b)

        res = np.dot(users, items.T)
        rmse = np.sqrt(((res * (test > 0) - test) ** 2).sum() / (test > 0).sum())
        print(rmse)

    return res


if __name__ == '__main__':
    data = load(Hyper.train)
    pred = linear(data)
    save_results(pred, Hyper.linear)
    create_submission(pred)
