# * Decision trees with some form of pruning
# * Boosting
# * Support Vector Machines
# * k-nearest neighbors
# * Neural networks
import time

from utils import *
from sklearn.datasets import fetch_openml, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer


def compare(X_train, X_test, y_train, y_test, models, dataset_name="", scorer=balanced_accuracy_score):
    for model_name, model in models.items():
        print(f"{model_name} on {dataset_name}")
        pipeline = build_pipeline(model)
        fig = plot_learning_curve(estimator=pipeline, title=f"Learning Curve - {model_name}", X=np.r_[X_train, X_test],
                                  y=np.r_[y_train, y_test], cv=None, scorer=scorer)
        fig.savefig(f'../output/{dataset_name}-{model_name}-Learning Curve.png')
        if hasattr(model, "partial_fit"):
            fig = plot_training_curve(pipeline, X_train, y_train, X_test, y_test, title="Performance over iterations",
                                      scorer=scorer,
                                      n_epoch=5, n_batch=min(128, X_train.shape[0]//4))
            fig.savefig(f'../output/{dataset_name}-{model_name}-Training Curve.png')


def calc_param_grid_decision_train(num_features):
    return {
            "criterion": ["gini", "entropy"],
            "max_depth": [5, 10, 15, int(np.sqrt(num_features)), int(np.log(num_features))],
            "max_features": ["sqrt", "log2"],
            "ccp_alpha": np.arange(0.01, 0.05, 0.005)
    }


param_grid_xgb = {"learning_rate": [0.001, 0.01], "n_estimators": [50, 100, 200, 500]}
param_grid_svm_sgd = {"loss": ['hinge']}
param_grid_svm = {"C": [0.5, 0.9, 1], "kernel": ["linear", "poly", "rbf", "sigmoid"]}
param_grid_knn = {"n_neighbors": [1, 5, 10, 50, 100]}
param_grid_nn = {"hidden_layer_sizes": [64, (4, 16), (8, 8), (16, 4), (4, 4, 4)]}

if __name__ == "__main__":
    scorer = make_scorer(balanced_accuracy_score)
    # MNIST
    print("===============================MNIST===============================")
    X, y = fetch_mnist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    num_features = X_train.shape[0]

    # Select model
    model_candidates = {
        "Decision Tree": (DecisionTreeClassifier, calc_param_grid_decision_train(num_features)),
        "XGBoosting": (XGBClassifier, param_grid_xgb),
        "Support Vector Machine": (SGDClassifier, param_grid_svm_sgd),
        "KNN": (KNeighborsClassifier, param_grid_knn),
        "Neural Network": (MLPClassifier, param_grid_nn)
    }
    models = {k: select_models(v[0], v[1], X_train, y_train) for k, v in model_candidates.items()}

    for model_name, model in models.items():
        print(model_name)
        print(model)
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        print("Training Time", end-start)

        print("Train:", scorer(model, X=X_train, y_true=y_train))
        start = time.time()
        print("Test:", scorer(model, X=X_test, y_true=y_test))
        end = time.time()
        print("Prediction Time", end - start)

    compare(X_train, X_test, y_train, y_test, models, dataset_name="MNIST", scorer=scorer)

    # Wine Quality
    print("===============================Wine Quality===============================")
    X, y = fetch_wine()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_features = X_train.shape[0]

    # Select model
    model_candidates = {
        "Decision Tree": (DecisionTreeClassifier, calc_param_grid_decision_train(num_features)),
        "XGBoosting": (XGBClassifier, param_grid_xgb),
        "Support Vector Machine": (SVC, param_grid_svm),
        "KNN": (KNeighborsClassifier, param_grid_knn),
        "Neural Network": (MLPClassifier, param_grid_nn)
    }
    models = {k: select_models(v[0], v[1], X_train, y_train, resampling=False) for k, v in model_candidates.items()}

    for model_name, model in models.items():
        print(model_name)
        print(model)
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        print("Training Time", end-start)

        print("Train:", scorer(model, X=X_train, y_true=y_train))
        start = time.time()
        print("Test:", scorer(model, X=X_test, y_true=y_test))
        end = time.time()
        print("Prediction Time", end - start)

    compare(X_train, X_test, y_train, y_test, models, dataset_name="Wine Quality", scorer=scorer)
