import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline

from numba import jit


@jit(nopython=True)
def delta_matrix(y):
    return (y[:, None] == y).astype(np.float32)


class SPCA(BaseEstimator, TransformerMixin):
    def __init__(self, k=2, eps=1e-3):
        super().__init__()
        self.k = k
        self.eps = eps
        self.U = None

    def fit(self, X, y):
        L = delta_matrix(y)
        Lc = L - L.mean(0)
        Q = X.T @ Lc @ X
        v, U = np.linalg.eigh(Q + self.eps*np.eye(len(Q)))
        self.U = U[:, ::-1]
        return self

    def transform(self, X):
        if self.U is None:
            raise RuntimeError("SPCA has not been fitted. Call 'fit' first.")
        reduced = X @ self.U[:, :self.k]
        return reduced


class KSPCA(BaseEstimator, TransformerMixin):
    def __init__(self, k=2, eps=1e-3, metric='rbf', gamma=1):
        super().__init__()
        self.k = k
        self.eps = eps
        self.metric = metric
        self.gamma = gamma
        self.orig_data, self.V = (None, None)

    def fit(self, X, y):
        self.train_data = X
        L = delta_matrix(y)
        K = pairwise_kernels(X, metric=self.metric, gamma=self.gamma)
        Lc = L - L.mean(0)
        Q = Lc @ K
        w, V = np.linalg.eig(Q + self.eps * np.eye(len(Q)))
        idx = np.argsort(w)
        V = V[:, idx].real
        self.V = V[:, ::-1]
        return self

    def transform(self, X):
        K = pairwise_kernels(X, self.train_data, metric=self.metric, gamma=self.gamma)
        reduced = K @ self.V[:, :self.k]
        return reduced


class SRP(BaseEstimator, TransformerMixin):
    def __init__(self, k=2, y_gamma=100, postproc=None):
        super().__init__()
        self.k = k
        self.y_gamma = y_gamma
        self.postproc = postproc
        self.y_features = RBFSampler(gamma=y_gamma, n_components=self.k)
        self.U = None

    def fit(self, X, y, center=True):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        psi = self.y_features.fit_transform(y).astype('float32')
        if self.postproc:
            psi = self.postproc(psi)
        if center:
            X = X - X.mean(0)
        self.U = psi.T @ X
        return self

    def transform(self, X):
        return X @ self.U.T


class KSRP(BaseEstimator, TransformerMixin):
    def __init__(self, k=2, gamma=1, n_components=500, metric='rbf', y_gamma=100,
                 x_postproc=None, y_postproc=None):
        super().__init__()
        self.k = k
        self.gamma = gamma
        self.n_components = n_components
        self.y_gamma = y_gamma
        self.metric = metric
        self.x_features = RBFSampler(gamma=gamma, n_components=n_components)
        self.y_features = RBFSampler(gamma=y_gamma, n_components=self.k)
        self.x_postproc = x_postproc
        self.y_postproc = y_postproc
        self.U = None
        self.psi = None
        self.train_data = None

    def fit(self, X, y, center=True):
        self.train_data = X
        phi = self.x_features.fit_transform(X).astype('float32')
        if self.x_postproc:
            phi = self.x_postproc(phi)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        psi = self.y_features.fit_transform(y).astype('float32')
        if psi.shape[1] > self.k:
            psi = psi[:, :self.k]
        if self.y_postproc:
            psi = self.y_postproc(psi)

        if center:
            phi = phi - phi.mean(0)
        self.psi = psi
        self.U = psi.T @ phi
        return self

    def transform(self, X, exact=False):
        # KSRP can be KSPCA with only the SPCA part approximated by SRP
        if exact:
            K = pairwise_kernels(X, self.train_data, metric=self.metric, gamma=self.gamma)
            return K @ self.psi
        # or both steps can be approximated with random features
        else:
            phi = self.x_features.transform(X).astype('float32')
            if self.x_postproc:
                phi = self.x_postproc(phi)
            return phi @ self.U.T


if __name__ == '__main__':
        import radMLBench
        #for dataset in radMLBench.listDatasets():
        dataset = "Song2020"
        X, y = radMLBench.loadData(dataset, return_X_y=True, local_cache_dir="./datasets")


        print ("Testing SPCA")
        spca = SPCA(k = 16)
        X.shape
        spca.fit(X, y)
        spca.transform(X).shape

        print ("Testing KSPCA")
        spca = KSPCA(k = 32)
        X.shape
        spca.fit(X, y)
        spca.transform(X).shape
        spca.fit_transform(X,y ).shape

        print ("Testing SRP")
        spca = SRP(k = 32)
        X.shape
        spca.fit(X, y)
        spca.transform(X).shape
        spca.fit_transform(X,y ).shape

        print ("Testing KSRP")
        spca = KSRP(k = 16)
        X.shape
        spca.fit(X, y)
        spca.transform(X).shape
        spca.fit_transform(X,y ).shape

#
