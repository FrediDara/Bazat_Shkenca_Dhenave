# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:52:16 2026

@author: Fredi
"""

import numpy as np
from sklearn.metrics import pairwise_distances


def kmedoids(X, n_clusters, metric="euclidean",
             max_iter=100, random_state=42):

    rng = np.random.default_rng(random_state)
    D = pairwise_distances(X, metric=metric)
    medoids = rng.choice(len(X), n_clusters, replace=False)

    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new_medoids = medoids.copy()

        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members):
                costs = D[np.ix_(members, members)].sum(axis=1)
                new_medoids[k] = members[np.argmin(costs)]

        if np.array_equal(medoids, new_medoids):
            break
        medoids = new_medoids

    labels = np.argmin(D[:, medoids], axis=1)
    return medoids, labels

