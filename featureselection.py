from scipy.stats import kendalltau, pearsonr
from scipy.stats import ttest_ind
from functools import partial
from sklearn.ensemble import RandomForestClassifier
import boruta

import cv2
import sys
sys.path.append("./pymrmre")
from pymrmre import *
import numpy as np


def boruta_fct (X, y):
    rfc = RandomForestClassifier(n_jobs=1, max_depth = 5, class_weight='balanced_subsample')
    b = boruta.BorutaPy (rfc, n_estimators = 'auto')
    b.fit(X, y)
    scores = np.max(b.ranking_) - b.ranking_
    return np.array(scores)



def ttest_score(X, y):
    scores = []
    for col in range(X.shape[1]):
        class_0 = X[y == 0, col]
        class_1 = X[y == 1, col]
        score, _ = ttest_ind(class_0, class_1, equal_var=False)
        scores.append(abs(score))
    return np.array(scores)


def bhattacharyya_score_fct (X, y):
    yn = y/np.sum(y)
    yn = np.asarray(yn, dtype = np.float32)
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        column = X[:, j]
        if np.min(column) == np.max(column):
            scores[j] = 0
        else:
            xn = (column - np.min(column)) / (np.max(column) - np.min(column))
            xn = xn / np.sum(xn)
            xn = np.asarray(xn, dtype = np.float32)
            scores[j] = cv2.compareHist(xn, yn, cv2.HISTCMP_BHATTACHARYYA)

    scores = np.asarray(scores, dtype = np.float32)
    return -scores


def kendall_score_func(X, y):
    scores = np.apply_along_axis(lambda x: abs(kendalltau(x, y)[0]), axis=0, arr=X)
    return scores.astype(np.float32)


def pearson_score_func(X, y):
    scores = np.apply_along_axis(lambda x: abs(pearsonr(x, y)[0]), axis=0, arr=X)
    return scores.astype(np.float32)


def _mrmr_score (X, y, nFeatures, nSolutions):
    Xp = pd.DataFrame(X, columns = range(X.shape[1]))
    yp = pd.DataFrame(y, columns=['Target']).astype(int)

    # we need to pre-specify the max solution length...
    solutions = mrmr.mrmr_ensemble(features = Xp, targets = yp, solution_length=nFeatures, solution_count=nSolutions)
    scores = [0]*Xp.shape[1]
    for k in solutions.iloc[0]:
        for j, z in enumerate(k):
            scores[z] = scores[z] + Xp.shape[1] - j
    scores = np.asarray(scores, dtype = np.float32)
    scores = scores/np.sum(scores)
    return scores


def mrmr_score (X, y, nFeatures):
    return _mrmr_score (X, y, nFeatures, 1)


def mrmre_score (X, y, nFeatures):
    return _mrmr_score (X, y, nFeatures, 5)


#
