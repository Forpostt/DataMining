# coding: utf-8

import numpy as np
from hyper import Hyper


def load(path):
    with open(path) as fd:
        data = fd.readlines()

    users, movies = set(), set()
    for record in data:
        user_id, movie_id, _ = record.strip().split('\t')
        users.add(int(user_id))
        movies.add(int(movie_id))

    train = np.zeros(shape=(max(users), max(movies)), dtype=np.uint8)
    for record in data:
        user_id, movie_id, score = record.strip().split('\t')
        train[int(user_id) - 1, int(movie_id) - 1] = float(score)

    return train


def create_submission(predicts):
    with open(Hyper.submission, 'w') as sub:
        sub.write('Id,Score\n')
        with open(Hyper.test) as fd:
            for i, line in enumerate(fd):
                user_id, movie_id = line.strip().split('\t')
                sub.write('{0},{1:.1f}\n'.format(i + 1, predicts[int(user_id) - 1, int(movie_id) - 1]))


def save_results(mm, path):
    with open(path, 'w') as fd:
        for i in range(mm.shape[0]):
            for j in range(mm.shape[1]):
                fd.write('{}\t{}\t{}\n'.format(i + 1, j + 1, mm[i, j]))
