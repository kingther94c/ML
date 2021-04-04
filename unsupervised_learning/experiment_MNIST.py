import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, make_scorer, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

from supervised_learning.utils import fetch_mnist, fetch_wine, build_pipeline
from unsupervised_learning.utils import clustering_k, dim_reduction_k, feature_selection_k, clustering_as_dr_k
from multiprocessing import Pool
from tqdm import tqdm
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, GridSearchCV
dataset_prefix = "MNIST"
dataset_dict = {}
X, y = fetch_mnist()
X = X/255
dataset_dict[dataset_prefix+"_original"] = train_test_split(X, y, test_size=0.2, random_state=13)
dataset_dict_exp5 = {}
ks_clusters = [2, 5, 10, 15]
ks_components = [2, 3, 5]
multiprocs = 1


def run_task_clustering(task):
    k, dataset, model_name, dataset_name = task
    X_train, X_test, y_train, y_test = dataset
    return clustering_k(k, X_train, y_train, model_name, dataset_name)


def run_task_dim_reduction(task):
    k, dataset, model_name, dataset_name = task
    X_train, X_test, y_train, y_test = dataset
    if model_name in {"RFE", "FSFS"}:
        res, model = feature_selection_k(k, X_train, y_train, model_name, dataset_name)
    else:
        res, model = dim_reduction_k(k, X_train, y_train, model_name, dataset_name)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    dataset_dict[dataset_prefix + f"_{model_name}({k})"] = X_train, X_test, y_train, y_test
    if k == 2:
        fig, ax = plt.subplots()
        ax.set_title(model_name + " in 2D")
        sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, ax=ax)
        plt.close(fig)
        fig.savefig(f"..//output//ul_{dataset_prefix}_experiment2_{model_name}_in2D.png")
    return res


def run_task_clustering_as_dr(task):
    k, dataset, model_name, dataset_name = task
    X_train, X_test, y_train, y_test = dataset
    res, transform = clustering_as_dr_k(k, X_train, y_train, model_name, dataset_name)
    X_train = transform(X_train)
    X_test = transform(X_test)
    dataset_dict_exp5[dataset_prefix + f"_{model_name}({k})"] = X_train, X_test, y_train, y_test
    if k == 2:
        fig, ax = plt.subplots()
        ax.set_title(model_name + " in 2D")
        sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, ax=ax)
        plt.close(fig)
        fig.savefig(f"..//output//ul_{dataset_prefix}_experiment5_{model_name}_in2D.png")
    return res

# Wine Quality Original
dataset = dataset_dict[dataset_prefix+"_original"]
X, y = fetch_wine()

print("EXP1")
results = []
tasks = [(k, dataset, model_name, dataset_prefix+"_original") for model_name in ("KMeans", "EM") for k in ks_clusters]
with Pool(multiprocs) as p:
    results = list(p.imap(run_task_clustering, tqdm(tasks)))
df = pd.DataFrame(results)
fig, axs = plt.subplots(1, 3, figsize=(16, 4))
axs[0].set_title("runtime vs k")
axs[0].set_xlabel("Runtime")
axs[1].set_title("adjusted_mutual_info_score vs k")
axs[1].set_xlabel("Adjusted Mutual Info Score")
axs[2].set_title("adjusted_rand_score vs k")
axs[2].set_xlabel("Adjusted Rand Score")
for model_name, df_ in df.groupby("model"):
    df_ = df_.set_index("k")
    df_["runtime"].plot(marker="x", ax=axs[0], label=model_name)
    df_["adjusted_mutual_info_score"].plot(marker="x", ax=axs[1], label=model_name)
    df_["adjusted_rand_score"].plot(marker="x", ax=axs[2], label=model_name)
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.close(fig)
fig.savefig(f"..//output//ul_{dataset_prefix}_experiment1_comparison_vs_k.png")
df.to_csv(f"..//output//ul_{dataset_prefix}_experiment1.csv")

# Dim Reduction
print("EXP2")
tasks = [(k, dataset, model_name, dataset_prefix+"_original") for model_name in ("PCA", "ICA", "RP", "RFE") for k in ks_components]

results = list(map(run_task_dim_reduction, tqdm(tasks)))
pd.DataFrame(results).to_csv(f"..//output//ul_{dataset_prefix}_experiment2.csv")


print("EXP3")
results = []
tasks = [(k, dataset, model_name, dataset_name) for model_name in ("KMeans", "EM") for k in ks_clusters for dataset_name, dataset in dataset_dict.items() if "original" not in dataset_name]
with Pool(multiprocs) as p:
    results = list(map(run_task_clustering, tqdm(tasks)))
df = pd.DataFrame(results)

df_selected = df.copy()
df_selected["DR"] = df_selected["dataset"].apply(lambda x: x.split("(")[0])
df_selected = df_selected.sort_values("adjusted_mutual_info_score")
df_selected = df_selected.groupby(["k", "model", "DR"]).last().reset_index()

fig, axs = plt.subplots(1, 3, figsize=(16, 4))
axs[0].set_title("avg runtime vs k")
axs[0].set_xlabel("Runtime")
axs[1].set_title("best adjusted_mutual_info_score vs k")
axs[1].set_xlabel("Adjusted Mutual Info Score")
axs[2].set_title("best adjusted_rand_score vs k")
axs[2].set_xlabel("Adjusted Rand Score")
for (model_name, dr), df_ in df_selected.groupby(["model", "DR"]):
    df_ = df_.set_index("k")
    df_["runtime"].plot(marker="x", ax=axs[0], label=model_name+"-"+dr)
    df_["adjusted_mutual_info_score"].plot(marker="x", ax=axs[1], label=model_name+"-"+dr)
    df_["adjusted_rand_score"].plot(marker="x", ax=axs[2], label=model_name+"-"+dr)
axs[2].legend()
plt.close(fig)
fig.savefig(f"..//output//ul_{dataset_prefix}_experiment3_comparison_vs_k.png")
df.to_csv(f"..//output//ul_{dataset_prefix}_experiment3.csv")

pd.DataFrame(results).to_csv(f"..//output//ul_{dataset_prefix}_experiment3.csv")

print("EXP4")
results = []
scorer = make_scorer(balanced_accuracy_score)
for dataset_name, dataset in dataset_dict.items():
    if "original" in dataset_name:
        continue
    X_train, X_test, y_train, y_test = dataset
    param_grid_nn = {"hidden_layer_sizes": [64, (4, 16), (8, 8), (16, 4), (4, 4, 4)]}
    param_grid_pipline = {"estimator__" + k: v for k, v in param_grid_nn.items()}
    pipline = build_pipeline(MLPClassifier(), resampling=False)
    gscv = GridSearchCV(pipline, param_grid_pipline, n_jobs=-3, verbose=1)
    res = gscv.fit(X_train, y_train)
    best_estimator = res.best_estimator_
    best_estimator.fit(X_train, y_train)
    train_score = scorer(best_estimator, X=X_train, y_true=y_train)
    test_score = scorer(best_estimator, X=X_test, y_true=y_test)
    dr = dataset_name.split("(")[0].split("_")[1]
    k = int(dataset_name.split("(")[1].split(")")[0])
    results.append({"dataset": dataset_name, "k": k, "DR": dr,"best_estimator": res.best_params_, "train_score": train_score, "test_score":test_score})
df = pd.DataFrame(results)
df.to_csv(f"..//output//ul_{dataset_prefix}_experiment4.csv")
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].set_title("Balanced Accuracy on Train dataset vs k (best NN)")
axs[0].set_xlabel("Balanced Accuracy on Train dataset")
axs[1].set_title("Balanced Accuracy on Test dataset vs k (best NN)")
axs[1].set_xlabel("Balanced Accuracy on Test dataset")
for dr, df_ in df.groupby( "DR"):
    df_ = df_.set_index("k")
    df_["train_score"].plot(marker="x", ax=axs[0], label=dr)
    df_["test_score"].plot(marker="x", ax=axs[1], label=dr)
axs[0].legend()
axs[1].legend()
plt.close(fig)
fig.savefig(f"..//output//ul_{dataset_prefix}_experiment4_comparison_vs_k.png")
df.to_csv(f"..//output//ul_{dataset_prefix}_experiment4.csv")

print("EXP5-1")
tasks = [(k, dataset, model_name, dataset_name) for model_name in ("KMeans", "EM") for k in ks_components for dataset_name, dataset in dataset_dict.items()]

results = list(map(run_task_clustering_as_dr, tqdm(tasks)))
pd.DataFrame(results).to_csv(f"..//output//ul_{dataset_prefix}_dr_experiment5-1.csv")

print("EXP5-2")
results = []
scorer = make_scorer(balanced_accuracy_score)
for dataset_name, dataset in dataset_dict_exp5.items():
    if "original" in dataset_name:
        continue
    X_train, X_test, y_train, y_test = dataset
    param_grid_nn = {"hidden_layer_sizes": [64, (4, 16), (8, 8), (16, 4), (4, 4, 4)]}
    param_grid_pipline = {"estimator__" + k: v for k, v in param_grid_nn.items()}
    pipline = build_pipeline(MLPClassifier(), resampling=False)
    gscv = GridSearchCV(pipline, param_grid_pipline, n_jobs=-3, verbose=1)
    res = gscv.fit(X_train, y_train)
    best_estimator = res.best_estimator_
    best_estimator.fit(X_train, y_train)
    train_score = scorer(best_estimator, X=X_train, y_true=y_train)
    test_score = scorer(best_estimator, X=X_test, y_true=y_test)
    dr = dataset_name.split("(")[0].split("_")[1]
    k = int(dataset_name.split("(")[1].split(")")[0])
    results.append({"dataset": dataset_name, "k": k, "DR": dr, "best_estimator": res.best_params_, "train_score": train_score, "test_score":test_score})
df = pd.DataFrame(results)
df.to_csv(f"..//output//ul_{dataset_prefix}_experiment5-2.csv")
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].set_title("Balanced Accuracy on Train dataset vs k (best NN)")
axs[0].set_xlabel("Balanced Accuracy on Train dataset")
axs[1].set_title("Balanced Accuracy on Test dataset vs k (best NN)")
axs[1].set_xlabel("Balanced Accuracy on Test dataset")
for dr, df_ in df.groupby(["DR"]):
    df_ = df_.set_index("k")
    df_["train_score"].plot(marker="x", ax=axs[0], label=dr)
    df_["test_score"].plot(marker="x", ax=axs[1], label=dr)
axs[0].legend()
axs[1].legend()
plt.close(fig)
fig.savefig(f"..//output//ul_{dataset_prefix}_experiment5_comparison_vs_k.png")