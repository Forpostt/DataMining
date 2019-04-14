# coding: utf-8

import numpy as np
from hyper import Hyper


def load():
    with open(Hyper.train) as fd:
        data = fd.readlines()

    users, movies = set(), set()
    for record in data:
        user_id, movie_id, _ = record.strip().split('\t')
        users.add(int(user_id))
        movies.add(int(movie_id))

    train = np.zeros(shape=(max(users), max(movies)), dtype=np.uint8)
    for record in data:
        user_id, movie_id, score = record.strip().split('\t')
        train[int(user_id) - 1, int(movie_id) - 1] = int(score)

    return train


def create_submission(predicts):
    with open(Hyper.submission, 'w') as sub:
        sub.write('Id,Score\n')
        with open(Hyper.test) as fd:
            for i, line in enumerate(fd):
                user_id, movie_id = line.strip().split('\t')
                sub.write('{0},{1:.1f}\n'.format(i + 1, predicts[int(user_id) - 1, int(movie_id) - 1]))


