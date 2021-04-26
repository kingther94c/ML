#!/usr/bin/env python
# coding: utf-8
import itertools
import math
import random
import numpy as np
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import time

eps = 1e-4


class MDP:
    def __init__(self):
        pass

    def encode_state(self, state):
        return self.state_map[state]

    def encode_action(self, action):
        return self.action_map[action]

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space

    def get_reward(self):
        return self.reward

    def prob(self, s, a):
        probs = self._prob(self.state_space[s], self.action_space[a])
        return {self.encode_state(next_state): prob for (next_state, prob) in probs.items()}

    def _prob(self, state, action):
        # {next_state: prob}
        pass

    def is_terminated(self, s):
        return self._is_terminated(self.state_space[s])

    def _is_terminated(self, state):
        pass

    def step(self, a):
        if self.is_terminated(self.s_curr):
            print("Reset!")
            return self.s_curr, 0, True

        s_next_probs = self.prob(self.s_curr, a)
        s = random.choices(list(s_next_probs.keys()), s_next_probs.values())[0]
        self.s_curr = s
        r = self.reward[self.s_curr]
        done = self.is_terminated(self.s_curr)
        return s, r, done

    def reset(self):
        self.s_curr = self.encode_state(self.start_state)
        return self.s_curr


class FrozonGridWorld(MDP):
    def __init__(self, amap="SHFFFHFHFFFHHFFG", p_drift=0.2, reward_map={"S": -1, "F": -1, "H": -1000, "G": 1000}):
        amap = np.asarray(list(amap))
        side = int(np.sqrt(amap.shape[0]))
        self.side = side
        self.amap = amap.reshape((side, side))
        self.state_space = [(i, j) for i in range(side) for j in range(side)]
        self.action_space = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.reward = [reward_map[self.amap[i, j]] for (i, j) in self.state_space]
        self.p_drift = p_drift
        self.p_forward = 1 - p_drift

        self.start_state = None
        for i, j in self.state_space:
            if self.amap[i, j] == "S":
                self.start_state = (i, j)
                break
        self.s_curr = None
        self.state_map = {state: s for s, state in enumerate(self.state_space)}
        self.action_map = {action: a for a, action in enumerate(self.action_space)}

    def _is_terminated(self, state):
        return self.amap[state[0], state[1]] in {"H", "G"}

    def _prob(self, state, action):
        x, y = state
        if self._is_terminated(state):
            return {}

        dx, dy = action
        if dx == 0:
            dft_x = 1
            dft_y = 0
        else:
            dft_x = 0
            dft_y = 1

        tmp_prob = {(x + dx, y + dy): self.p_forward,
                    (x + dft_x, y + dft_y): self.p_drift / 2,
                    (x - dft_x, y - dft_y): self.p_drift / 2}

        state_probs = {}
        for (nx, ny), p in tmp_prob.items():
            if nx >= self.side or nx < 0:
                nx = x
            if ny >= self.side or ny < 0:
                ny = y
            state_probs[(nx, ny)] = state_probs.get((nx, ny), 0) + p

        return state_probs


class SnakesLaddersGame(MDP):
    def __init__(self, size=10000, num_ladder=1000, num_dice=2, random_seed=42, additional_ladders={}):
        self.size = size
        self.state_space = [i for i in range(size)]
        self.action_space = [i for i in range(num_dice+1)]
        self.reward = [-1 if i < size - 1 else 100 for i in self.state_space]
        self.dice_probs = [self.calc_dice_prob(action) for action in self.action_space]

        np.random.seed(random_seed)
        heads = np.random.choice(self.state_space, size=num_ladder, replace=False)
        tails = np.random.choice(self.state_space, size=num_ladder, replace=True)
        self.ladders = dict(zip(heads, tails))
        self.ladders.update(additional_ladders)

        self.start_state = 0
        self.s_curr = 0
        self.state_map = {state: s for s, state in enumerate(self.state_space)}
        self.action_map = {action: a for a, action in enumerate(self.action_space)}

    def calc_dice_prob(self, num_dice):
        if num_dice == 0:
            return {1: 1}
        probs_sum = {}
        probs_single = {m: 1 / 6 for m in range(1, 7)}
        for ks in itertools.product(probs_single, repeat=num_dice):
            v = sum(ks)
            p = math.prod(probs_single[k] for k in ks)
            probs_sum[v] = probs_sum.get(v, 0) + p
        return probs_sum

    def _is_terminated(self, state):
        return state >= self.size - 1

    def _prob(self, state, action):
        if self._is_terminated(state):
            return {}
        tmp_prob = self.dice_probs[action]
        state_probs = {}
        for move, p in tmp_prob.items():
            next_state = self.ladders.get(state + move, state + move)
            if next_state >= self.size - 1:
                next_state = self.size - 1
            state_probs[next_state] = state_probs.get(next_state, 0) + p
        return state_probs


def policy_valuation(pi, env, gamma=0.99):
    r = env.reward
    v = np.ones(len(env.state_space))
    v_new = np.ones(len(env.state_space))
    dv = 1000*eps
    while dv > eps:
        for s in range(len(v)):
            v_new[s] = sum(r[s_next]*p +v[s_next]*p*gamma for s_next, p in env.prob(s, pi[s]).items())
        dv = np.abs(v_new - v).max()
        v = v_new.copy()
    return v


def policy_improvement(v, env, gamma=0.99):
    r = env.reward
    pi_new = [np.argmax([sum(r[s_next]*p +v[s_next]*p*gamma for s_next, p in env.prob(s, a).items()) for a in range(len(env.action_space))]) for s in range(len(v))]
    return pi_new


def encode_policy(pi):
    pi_encoded = "".join([str(a) for a in pi])
    return pi_encoded


direction_map = {
    0: '↓',
    1: '↑',
    2: '→',
    3: '←',
}


def plot_policy_map_frozon_lake(title, pi, env, direction_map=direction_map):
    map_desc = env.amap
    policy = np.array(pi).reshape(map_desc.shape)
    color_map = {
        'S': 'green',
        'F': 'skyblue',
        'H': 'black',
        'G': 'gold',
    }

    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            text = ax.text(x + 0.5, y + 0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                   path_effects.Normal()])

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()

    return fig


# Policy Iteration
def policy_iteration(env, gamma=0.99, iter_limit=100000):
    history = []
    pi_new = [random.randint(0, len(env.action_space) - 1) for i in range(len(env.state_space))]
    v_new = policy_valuation(pi_new, env, gamma)
    dv = 100*eps
    i = 0
    clock = 0
    while dv > eps and i < iter_limit:
        pi = pi_new
        v = v_new
        start = time.process_time()
        pi_new = policy_improvement(v, env, gamma)
        v_new = policy_valuation(pi_new, env, gamma)
        clock += time.process_time() - start
        dv = np.abs(v_new - v).max()
        rec = {"algo": f"PI({gamma})", "i":i, "ER":policy_valuation(pi, env, gamma=.99)[0], "dv": dv, "v": v, "pi": pi, "time": clock}
        print(rec["i"], rec["ER"], rec["dv"])
        history.append(rec)
        i += 1
    return v, pi, history


# Value Iteration
def value_iteration(env, gamma=0.99, iter_limit=100000):
    history = []
    r = env.reward
    v = np.random.rand(len(env.state_space))
    v_new = np.random.rand(len(env.state_space))
    dv = 1000*eps
    i = 0
    pi = None
    clock = 0
    while dv > eps and i < iter_limit:
        start = time.process_time()
        for s in range(len(v)):
            v_new[s] = max(sum(r[s_next]*p +v[s_next]*p*gamma for s_next, p in env.prob(s, a).items()) for a in range(len(env.action_space)))
        clock += time.process_time() - start
        dv = np.abs(v_new - v).max()
        pi = policy_improvement(v, env, gamma=1)
        rec = {"algo": f"VI({gamma})", "i":i, "ER":policy_valuation(pi, env, gamma=.99)[0], "dv": dv, "v": v, "time": clock}
        print(rec["i"], rec["ER"], rec["dv"])
        history.append(rec)
        v = v_new.copy()
        i += 1
    pi = policy_improvement(v, env, gamma)
    return v, pi, history


# Q Learning
def eps_greedy(s, q_table, epsilon, env):
    if np.random.random() < epsilon:  # explore
        a = np.random.randint(len(env.action_space))
    else:  # exploit
        a = np.argmax(q_table[s])
    return a


def q_learning(env, gamma=0.99, epsilon_start=0.99, epsilon_decay=0.999, epsilon_min=0.01, alpha=0.1, iter_limit=1000):
    epsilon = epsilon_start
    history = []
    clock = 0
    q_table = np.random.random((len(env.state_space), len(env.action_space)))
    s = env.reset()
    done = False

    dr = 1000 * eps
    i = 0
    reward_history = []
    reward_cumsum = 0
    pi = None
    start = time.process_time()
    while i < iter_limit:

        if done:
            clock += time.process_time() - start
            reward_history.append(reward_cumsum)
            reward_cumsum = 0
            s = env.reset()
            i += 1
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if i % 10 == 0 and i > 9:
                dr = abs(sum(reward_history[-10:]) / 10 - sum(reward_history[-20:-10]) / 10)
                pi = [np.argmax(q_table[s]) for s in range(len(env.state_space))]
                rec = {"algo": f"Q({gamma},{epsilon_decay},{alpha})", "i": i, "ER": policy_valuation(pi, env, gamma=.99)[0], "cumsum_reward": (sum(reward_history[-10:]) / 10),
                       "dv": dr, "epsilon": epsilon, "time": clock}
                print(rec)
                history.append(rec)
            start = time.process_time()
        a = eps_greedy(s, q_table, epsilon=epsilon, env=env)
        s_next, r, done = env.step(a)
        reward_cumsum += r
        if done:
            q_table[s, a] += alpha * (r - q_table[s, a])
        else:
            q_table[s, a] += alpha * (
                    r + gamma * max(q_table[s_next, a_next] for a_next in range(len(env.action_space))) - q_table[s, a])
        s = s_next
    return pi, q_table, history



