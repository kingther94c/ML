# * Decision trees with some form of pruning
# * Boosting
# * Support Vector Machines
# * k-nearest neighbors
# * Neural networks

import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml, load_wine
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score


def fetch_mnist():
    return fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)


def fetch_wine():
    wine_red = "../dataset/winequality-red.csv"
    data_red = pd.read_csv(wine_red, delimiter=",")
    data_red["is_red"] = 1
    wine_white = "../dataset/winequality-white.csv"
    data_white = pd.read_csv(wine_white, delimiter=",")
    data_white["is_red"] = 0

    data = pd.concat([data_red, data_white])
    y = data.quality.apply(lambda x: 0 if x == 5 else (1 if x >5 else -1))
    X = data.iloc[:, :-1]
    return X, y


def build_pipeline(estimator, resampling=True):
    if resampling:
        from imblearn.pipeline import Pipeline
        pipeline = Pipeline(steps=[("resampler", SMOTE()),
                                   ('pre_processor', StandardScaler()),
                                   ('estimator', estimator)])
    else:
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(steps=[('pre_processor', StandardScaler()),
                                   ('estimator', estimator)])

    return pipeline


def plot_learning_curve(estimator, X, y, title="Learning Curves", ylim=None, cv=None, scorer=None,
                        n_jobs=-3, train_sizes=np.linspace(.1, 1.0, 5)):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                                          train_sizes=train_sizes, verbose=1,
                                                                          scoring=scorer,
                                                                          return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    plt.close(fig)

    return fig


def plot_training_curve(model, X_train, y_train, X_test, y_test, title="Performance over iterations",
                        scorer=None,
                        n_epoch=5,
                        n_batch=128):
    """ Built based on sklearn's docs """
    n_samples = y_train.shape[0]
    n_classes = np.unique(y_train)

    scores_train = []
    scores_test = []

    if hasattr(model, "named_steps"):
        estimator = model.named_steps.estimator
        if "resampler" in model.named_steps:
            resampler = model.named_steps.resampler
            X_res, y_train = resampler.fit_resample(X_train, y_train)
            pre_processor = model.named_steps.pre_processor
            X_train = pre_processor.fit_transform(X_res)
            X_test = pre_processor.transform(X_test)
        else:
            pre_processor = model.named_steps.pre_processor
            X_train = pre_processor.fit_transform(X_train)
            X_test = pre_processor.transform(X_test)
    else:
        estimator = model

    # EPOCH
    epoch = 0
    while epoch < n_epoch:
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
            estimator.partial_fit(X_train[indices], y_train[indices], classes=n_classes)
            mini_batch_index += n_batch

            if mini_batch_index >= n_samples:
                break

            # SCORE TRAIN & TEST
            if scorer:
                scores_train.append(scorer(estimator=estimator, X=X_train, y_true=y_train))
                scores_test.append(scorer(estimator=estimator, X=X_test, y_true=y_test))
            else:
                scores_train.append(estimator.score(X=X_train, y=y_train))
                scores_test.append(estimator.score(X=X_test, y=y_test))
        epoch += 1

    """ Plot """
    fig, ax = plt.subplots(1, sharex=True, sharey=True)
    ax.grid()
    ax.plot(scores_train, alpha=0.8, label='Train')
    ax.plot(scores_test, alpha=0.8, label='Test')
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.legend(loc='upper left')

    fig.suptitle("Accuracy over epochs", fontsize=14)
    plt.close(fig)
    return fig


def select_models(Classifier, param_grid, X_train, y_train, resampling=True):
    param_grid_pipline = {"estimator__"+k: v for k, v in param_grid.items()}
    pipline = build_pipeline(Classifier(), resampling=resampling)
    gscv = GridSearchCV(pipline, param_grid_pipline, n_jobs=-3, verbose=1)
    res = gscv.fit(X_train, y_train)
    return res.best_estimator_

