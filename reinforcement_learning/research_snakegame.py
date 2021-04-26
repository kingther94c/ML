import numpy as np
import pandas
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

from reinforcement_learning.utils import SnakesLaddersGame, encode_policy, policy_valuation, policy_improvement, \
    FrozonGridWorld, eps, policy_iteration, plot_policy_map_frozon_lake, value_iteration, q_learning


def make_env():
    return SnakesLaddersGame(size=1000, num_ladder=30, num_dice=2, random_seed=99, additional_ladders={10: 990})


def run_one(task):
    env = make_env()
    gamma, eps_decay, alpha = task
    pi, q_table, history = q_learning(env, gamma=gamma, epsilon_start=0.99, epsilon_decay=eps_decay, epsilon_min=0.01,
                                      alpha=alpha, iter_limit=3000)
    return history


def pi(gamma):
    v, pi, history = policy_iteration(env, gamma=gamma)
    return history


def vi(gamma):
    v, pi, history = value_iteration(env, gamma=gamma)
    return history


env = make_env()
records = []
gammas = [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]

with Pool(5) as p:
    print("PI")
    for history in tqdm(p.imap(pi, gammas), total=len(gammas)):
        records.extend(history)

    print("VI")
    for history in tqdm(p.imap(vi, gammas), total=len(gammas)):
        records.extend(history)
df = pd.DataFrame(records)
df.to_csv(f"..//output//records_SnakesLaddersGame.csv")
df.to_pickle(f"..//output//records_SnakesLaddersGame.pickle")


tasks = []
for gamma in [0.7, 0.9, 0.99]:
    tasks.append((gamma, .995,  0.05))
for eps_decay in [0.99, .995, 0.999]:
    tasks.append((0.99, eps_decay, 0.05))
for alpha in [0.01, 0.05, 0.1]:
    tasks.append((0.99, .995, alpha))
tasks = sorted(set(tasks))
with Pool(7) as p:
    for history in tqdm(p.imap(run_one, tasks), total=len(tasks)):
        records.extend(history)

df = pd.DataFrame(records)
df.to_csv(f"..//output//records_SnakesLaddersGame.csv")
df.to_pickle(f"..//output//records_SnakesLaddersGame.pickle")
