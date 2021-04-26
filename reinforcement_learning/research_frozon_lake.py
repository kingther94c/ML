import numpy as np
import pandas
import pandas as pd

from reinforcement_learning.utils import SnakesLaddersGame, encode_policy, policy_valuation, policy_improvement, \
    FrozonGridWorld, eps, policy_iteration, plot_policy_map_frozon_lake, value_iteration, q_learning

amap = "".join([
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ])

env = FrozonGridWorld(amap=amap, p_drift=0.2, reward_map={"S": -1, "F": -1, "H": -1000, "G": 1000})
fig = plot_policy_map_frozon_lake("Frozon Lake (8 x 8)", [0 for s in env.state_space], env, direction_map={0: '', 1: '', 2: '', 3: ''})
fig.savefig("..//output//rl_frozon_lake.png")

records = []
for gamma in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
    print(f"Gamma = {gamma}")
    v, pi, history = policy_iteration(env, gamma=gamma)
    fig = plot_policy_map_frozon_lake(f"Frozon Lake (8 x 8) Policy Iteration ($\gamma={gamma}$)", pi, env)
    fig.savefig(f"..//output//rl_FrozonGridWorld_PI_{gamma}.png")
    records.extend(history)

    v, pi, history = value_iteration(env, gamma=gamma)
    fig = plot_policy_map_frozon_lake(f"Frozon Lake (8 x 8) Value Iteration ($\gamma={gamma}$)", pi, env)
    fig.savefig(f"..//output//rl_FrozonGridWorld_VI_{gamma}.png")
    records.extend(history)

df = pd.DataFrame(records)
df.to_csv(f"..//output//records_rozonGridWorld.csv")
df.to_pickle(f"..//output//records_FrozonGridWorld.pickle")

eps_decay_str = r"{df}_{\varepsilon}"


tasks = []
for gamma in [0.7, 0.9, 0.99]:
    tasks.append((gamma, .995,  0.05))
for eps_decay in [0.99, .995, 0.999]:
    tasks.append((0.99, eps_decay, 0.05))
for alpha in [0.01, 0.05, 0.1]:
    tasks.append((0.99, .995, alpha))
tasks = sorted(set(tasks))

for gamma, eps_decay, alpha in tasks:
    print(f"Q_{gamma}_{eps_decay}_{alpha}")
    pi, q_table, history = q_learning(env, gamma=gamma, epsilon_start=0.99, epsilon_decay=eps_decay, epsilon_min=0.01, alpha=alpha, iter_limit=1000)
    fig = plot_policy_map_frozon_lake(fr"Frozon Lake (8 x 8) Q Learning ($\gamma={gamma}, {eps_decay_str}={eps_decay}, \alpha={alpha}$)", pi, env)
    fig.savefig(f"..//output//rl_FrozonGridWorld_Q_{gamma}_{eps_decay}_{alpha}.png")
    records.extend(history)

df = pd.DataFrame(records)
df.to_csv(f"..//output//records_FrozonGridWorld.csv")
df.to_pickle(f"..//output//records_FrozonGridWorld.pickle")
