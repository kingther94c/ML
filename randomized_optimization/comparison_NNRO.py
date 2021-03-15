
from supervised_learning.comparison import *
from supervised_learning.utils import *
import mlrose
import time

from sklearn.preprocessing import OneHotEncoder


class NeuralNetwork(mlrose.NeuralNetwork):
    def __init__(self, hidden_nodes=None,
                 activation='relu',
                 algorithm='random_hill_climb',
                 max_iters=100,
                 bias=True,
                 is_classifier=True,
                 learning_rate=0.1,
                 early_stopping=False,
                 clip_max=1e+10,
                 restarts=0,
                 schedule=mlrose.GeomDecay(),
                 pop_size=200,
                 mutation_prob=0.1,
                 max_attempts=10,
                 random_state=None,
                 curve=False):
        super(NeuralNetwork, self).__init__(
            hidden_nodes=hidden_nodes,
            activation=activation,
            algorithm=algorithm,
            max_iters=max_iters,
            bias=bias,
            is_classifier=is_classifier,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            clip_max=clip_max,
            restarts=restarts,
            schedule=schedule,
            pop_size=pop_size,
            mutation_prob=mutation_prob,
            max_attempts=max_attempts,
            random_state=random_state,
            curve=curve)
        self.one_hot = OneHotEncoder(categories='auto')

    def fit(self,  X, y):
        y_trans = self.one_hot.fit_transform(y.reshape(-1, 1)).todense()
        super(NeuralNetwork, self).fit(X, y_trans)

    def predict(self, X):
        y_pred = super(NeuralNetwork, self).predict(X)
        return self.one_hot.inverse_transform(y_pred)


if __name__ == "__main__":
    scorer = make_scorer(balanced_accuracy_score)
    # # MNIST
    # print("===============================MNIST===============================")
    # X, y = fetch_mnist()
    # one_hot = OneHotEncoder(categories='auto')
    # y = one_hot.fit_transform(y.reshape(-1, 1)).todense()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    # num_features = X_train.shape[0]
    #
    # # Select model
    # model_candidates = {
    #     "Gradient Descent": (mlrose.NeuralNetwork, {"hidden_nodes": [[64]]}),
    #     "RHC": (mlrose.NeuralNetwork, {"hidden_nodes": [[64]],
    #                                    "restarts": [100]}),
    #     "SA": (mlrose.NeuralNetwork, {"hidden_nodes": [[64]],
    #                                   "schedule": [mlrose.GeomDecay(init_temp=10.0, decay=0.99, min_temp=0.001),
    #                                                mlrose.GeomDecay(init_temp=10.0, decay=0.9, min_temp=0.001),
    #                                                mlrose.GeomDecay(init_temp=10.0, decay=0.8, min_temp=0.001)]}),
    #     "GA": (mlrose.NeuralNetwork, {"hidden_nodes": [[64]],
    #                                   "pop_size": [300],
    #                                   "mutation_prob": [0.1, 0.3, 0.5, 0.7]}),
    # }
    # models = {k: select_models(v[0], v[1], X_train, y_train) for k, v in model_candidates.items()}
    #
    # for model_name, model in models.items():
    #     print(model_name)
    #     print(model)
    #     start = time.time()
    #     model.fit(X_train, y_train)
    #     end = time.time()
    #     print("Training Time", end-start)
    #
    #     print("Train:", scorer(model, X=X_train, y_true=y_train))
    #     start = time.time()
    #     print("Test:", scorer(model, X=X_test, y_true=y_test))
    #     end = time.time()
    #     print("Prediction Time", end - start)
    #
    # compare(X_train, X_test, y_train, y_test, models, dataset_name="MNIST", scorer=scorer, n_jobs=-3)


    print("===============================Wine Quality===============================")
    X, y = fetch_wine()
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
    num_features = X_train.shape[0]

    # Select model
    model_candidates = {
        "Gradient Descent": (NeuralNetwork, {"hidden_nodes": [[4, 16]]}),
        "RHC": (NeuralNetwork, {"hidden_nodes": [[4, 16]],
                                       "restarts": [100]}),
        "SA": (NeuralNetwork, {"hidden_nodes": [[4, 16]],
                                      "schedule": [mlrose.GeomDecay(init_temp=10.0, decay=0.99, min_temp=0.001),
                                                   mlrose.GeomDecay(init_temp=10.0, decay=0.9, min_temp=0.001),
                                                   mlrose.GeomDecay(init_temp=10.0, decay=0.8, min_temp=0.001)]}),
        "GA": (NeuralNetwork, {"hidden_nodes": [[4, 16]],
                                      "pop_size": [300],
                                      "mutation_prob": [0.1, 0.3, 0.5, 0.7]}),
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

    compare(X_train, X_test, y_train, y_test, models, dataset_name="Wine Quality", scorer=scorer, n_jobs=-3)
