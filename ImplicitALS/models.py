# coding: utf-8

import numpy as np
import tqdm
from sklearn.metrics.pairwise import cosine_similarity

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


if __name__ == '__main__':
    data = load()
    pred = item_based(data)
    save_results(pred, Hyper.item_based)
    create_submission(pred)
