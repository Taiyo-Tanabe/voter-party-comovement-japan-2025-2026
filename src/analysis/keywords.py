import numpy as np


def fightin_words(X, y, alpha0=1000):
    """Monroe, Colaresi & Quinn (2008) の Fightin' Words。語ごとの z 値を返す"""
    counts_i = np.asarray(X[y].sum(axis=0)).ravel()
    counts_j = np.asarray(X[~y].sum(axis=0)).ravel()
    total = counts_i + counts_j
    alpha = alpha0 * total / total.sum()
    n_i, n_j, a0 = counts_i.sum(), counts_j.sum(), alpha.sum()
    li = np.log((counts_i + alpha) / (n_i + a0 - counts_i - alpha))
    lj = np.log((counts_j + alpha) / (n_j + a0 - counts_j - alpha))
    return (li - lj) / np.sqrt(1 / (counts_i + alpha) + 1 / (counts_j + alpha))


def top_terms(X, terms, labels, k=15):
    """グループごとに、z 値の高い語を k 個ずつ返す"""
    labels = np.asarray(labels)
    return {g: list(terms[np.argsort(fightin_words(X, labels == g))[::-1][:k]])
            for g in np.unique(labels)}