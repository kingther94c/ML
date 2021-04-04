import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from supervised_learning.utils import fetch_mnist, fetch_wine
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class InvertibleRandomProjection(GaussianRandomProjection):
    """Gaussian random projection with an inverse transform using the pseudoinverse."""

    def __init__(
        self, n_components="auto", eps=0.3, orthogonalize=False, random_state=None
    ):
        self.orthogonalize = orthogonalize
        super().__init__(n_components=n_components, eps=eps, random_state=random_state)

    @property
    def pseudoinverse(self):
        """Pseudoinverse of the random projection.

        This inverts the projection operation for any vector in the span of the
        random projection. For small enough `eps`, this should be close to the
        correct inverse.
        """
        try:
            return self._pseudoinverse
        except AttributeError:
            if self.orthogonalize:
                # orthogonal matrix: inverse is just its transpose
                self._pseudoinverse = self.components_
            else:
                self._pseudoinverse = np.linalg.pinv(self.components_.T)
            return self._pseudoinverse

    def fit(self, X):
        super().fit(X)
        if self.orthogonalize:
            Q, _ = np.linalg.qr(self.components_.T)
            self.components_ = Q.T
        return self


    def inverse_transform(self, X):
        return X.dot(self.pseudoinverse)


def clustering_k(k, X, y, model_name, dataset_name):
    if model_name == "KMeans":
        model = KMeans(n_clusters=k)
    else:
        model = GaussianMixture(n_components=k)

    start = time.process_time()
    model.fit(X)
    end = time.process_time()
    run_time = end - start
    y_pred = model.predict(X)
    y_act = y

    amis = adjusted_mutual_info_score(y_pred, y_act)
    ars = adjusted_rand_score(y_pred, y_act)
    return {"dataset": dataset_name, "model": model_name, "k": k,
            "runtime": run_time, "adjusted_mutual_info_score": amis, "adjusted_rand_score": ars}


def clustering_as_dr_k(k, X, y, model_name, dataset_name):
    if model_name == "KMeans":
        model = KMeans(n_clusters=k)
        transform = model.transform
    else:
        model = GaussianMixture(n_components=k)
        transform = model.predict_proba

    start = time.process_time()
    model.fit(X)
    end = time.process_time()
    runtime = end - start
    res = {"dataset": dataset_name, "model": model_name, "k": k,
            "runtime": runtime}

    return res, transform


def dim_reduction_k(k, X, y, model_name, dataset_name):
    if model_name == "PCA":
        model = PCA(n_components=k)
    elif model_name == "ICA":
        model = FastICA(n_components=k)
    elif model_name == "RP":
        model = InvertibleRandomProjection(n_components=k)
    # elif model_name == "RFE":
    #     model = RFE(n_features_to_select=k, estimator=DecisionTreeClassifier())

    start = time.process_time()
    X_trans = model.fit_transform(X)
    end = time.process_time()
    runtime = end - start
    X_re = model.inverse_transform(X_trans)
    error = ((X_re - X)**2).mean().mean()
    res = {"dataset": dataset_name, "model": model_name, "k": k,
            "runtime": runtime, "reconstruction_error": error}
    if model_name == "PCA":
        res["explained_variance_ratio_"] = sum(model.explained_variance_ratio_)

    return res, model


def feature_selection_k(k, X, y, model_name=None, dataset_name=None):
    if model_name == "RFE":
        model = RFE(n_features_to_select=k, estimator=DecisionTreeClassifier())
    elif model_name == "FSFS":
        model = SequentialFeatureSelector(n_features_to_select=k, estimator=DecisionTreeClassifier(), direction="forward")

    start = time.process_time()
    X_trans = model.fit_transform(X, y)
    end = time.process_time()
    runtime = end - start
    X_re = model.inverse_transform(X_trans)
    error = ((X_re - X)**2).mean().mean()
    return {"dataset": dataset_name, "model": model_name, "k": k,
            "runtime": runtime, "reconstruction_error": error}, model
