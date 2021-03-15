import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

algos={mlrose.random_hill_climb:[{"restarts": 20}],
       mlrose.simulated_annealing:[{"schedule": mlrose.GeomDecay(init_temp=10.0, decay=0.9999, min_temp=0.001)},
                                   {"schedule": mlrose.GeomDecay(init_temp=10.0, decay=0.999, min_temp=0.001)},
                                   {"schedule": mlrose.GeomDecay(init_temp=10.0, decay=0.99, min_temp=0.001)},
                                   {"schedule": mlrose.GeomDecay(init_temp=10.0, decay=0.9, min_temp=0.001)},
                                   {"schedule": mlrose.GeomDecay(init_temp=10.0, decay=0.8, min_temp=0.001)},
                                   {"schedule": mlrose.GeomDecay(init_temp=10.0, decay=0.7, min_temp=0.001)},
                                   {"schedule": mlrose.GeomDecay(init_temp=10.0, decay=0.6, min_temp=0.001)},
                                   {"schedule": mlrose.GeomDecay(init_temp=10.0, decay=0.5, min_temp=0.001)},
                                   {"schedule": mlrose.ArithDecay(init_temp=10.0, decay=0.0001, min_temp=0.001)},
                                   {"schedule": mlrose.ArithDecay(init_temp=10.0, decay=0.001, min_temp=0.001)},
                                   {"schedule": mlrose.ArithDecay(init_temp=10.0, decay=0.01, min_temp=0.001)}
                                   ],
       mlrose.genetic_alg:[{"mutation_prob": .15},
                           {"mutation_prob": .35},
                           {"mutation_prob": .55},
                           {"mutation_prob": .75},
                           {"mutation_prob": .95}],
       mlrose.mimic:[{"keep_pct": .1},
                     {"keep_pct": .2},
                     {"keep_pct": .4},
                     {"keep_pct": .6},
                     {"keep_pct": .8},
                    ]
       }


def run_one(fitness_fn, nbit, algos):
    records = []
    # Initilize problem
    opt_prob = mlrose.DiscreteOpt(length=nbit, fitness_fn=fitness_fn)

    for algo, parameters in algos.items():
        for parameter in tqdm(parameters, desc=algo.__name__):
            # Track runtime
            start = time.process_time()
            if algo == mlrose.genetic_alg:
                best_state, best_fitness, fitness_curve = algo(problem=opt_prob,
                                                               max_attempts=10,
                                                               curve=True,
                                                               pop_size=10 * nbit,
                                                               **parameter)
            elif algo == mlrose.mimic:
                best_state, best_fitness, fitness_curve = algo(problem=opt_prob,
                                                               max_attempts=10,
                                                               curve=True,
                                                               pop_size=10 * nbit,
                                                               fast_mimic=True,
                                                               **parameter)

            else:
                best_state, best_fitness, fitness_curve = algo(problem=opt_prob, max_attempts=10, curve=True,
                                                               **parameter)
            end = time.process_time()
            runtime = end - start

            # Add record
            records.append({"nbit": nbit,
                            "algorithm": algo.__name__,
                            "parameter_dict": parameter,
                            "parameter": parameter_str(parameter),
                            "runtime": runtime,
                            "best_fitness": best_fitness,
                            "fitness_curve": fitness_curve})

    return records


def prod_consec_one(state):
    prod = 1
    count = 0
    for s in state:
        if s == 0 and count > 0:
            prod *= count
            count = 0
        elif s == 1:
            count += 1
    if count > 0:
        prod *= count
    return prod

def func_convert_bin_swap(state):
    n = len(state)//2
    a = int("".join([str(int(s)) for s in state[:n]]),2)
    b = int("".join([str(int(s)) for s in reversed(state[n:])]),2)
    return (a/(b+1))


def parameter_str(parameter):
    param = {**parameter}
    if "schedule" in parameter:
        if isinstance(parameter["schedule"], mlrose.GeomDecay):
            param["schedule"] = f"GeomDecay(decay={parameter['schedule'].decay})"
        elif isinstance(parameter["schedule"], mlrose.ArithDecay):
            param["schedule"] = f"ArithDecay(decay={parameter['schedule'].decay})"
    return ";".join(["=".join([f"{e}" for e in kv]) for kv in param.items()])


def plot_fitting_curve(records):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    for ax, algorithm, algo_name in zip(axs.reshape(-1),
                                        ["random_hill_climb", 'simulated_annealing', 'genetic_alg', 'mimic'],
                                        ["Random Hill Climb", 'Simulated Annealing', 'Genetic Algorithm', 'MIMIC']):
        recs = [rec for rec in records if rec["algorithm"] == algorithm]
        parameters = [parameter_str(rec["parameter_dict"]) for rec in recs]
        fitness_curves = [rec["fitness_curve"] for rec in recs]
        ax.set_title(algo_name)

        if max([rec["best_fitness"] for rec in recs]) > 1000000:
            ax.set_yscale("log")

        for p, fc in zip(parameters, fitness_curves):
            ax.plot(fc, label=p)
        ax.set_xlabel("iteration")
        ax.set_ylabel("fitness")
        ax.legend()

    plt.close(fig)
    return fig


def plot_vs_nbit(records):
    stats = pd.DataFrame(records).sort_values("best_fitness").groupby(["nbit", "algorithm"])[
        "best_fitness", "runtime"].last()

    rename = dict(zip(["random_hill_climb", 'simulated_annealing', 'genetic_alg', 'mimic'],
                      ["Random Hill Climb", 'Simulated Annealing', 'Genetic Algorithm', 'MIMIC']))
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].set_title("Runtime on Best Fitness")
    stats["runtime"].unstack().plot(ax=axs[0])
    axs[0].set_ylabel("runtime (sec)")
    axs[0].legend()

    axs[1].set_title("Best Fitness")
    best_fitness = stats["best_fitness"].unstack()
    if (best_fitness.max().max() > 1000000):
        axs[1].set_ylabel(r"$\frac{fitness}{10^{N}}$")
        best_fitness.apply(lambda x: x / np.power(10, x.name / 10), axis=1).plot(ax=axs[1])
    else:
        best_fitness.plot(ax=axs[1])
        axs[1].set_ylabel("fitness")
    axs[1].legend()

    return fig


continuous_peaks = mlrose.ContinuousPeaks(t_pct=0.1)
six_peaks = mlrose.SixPeaks(t_pct=0.1)
flip_flop = mlrose.FlipFlop()
product_consec_ones = mlrose.CustomFitness(fitness_fn=prod_consec_one, problem_type="discrete")
count_ones = mlrose.CustomFitness(fitness_fn=lambda state: sum(state), problem_type="discrete")
convert_bin_swap = mlrose.CustomFitness(fitness_fn=func_convert_bin_swap, problem_type="discrete")


if __name__ == "__main__":
    records_by_prob = {six_peaks: [], flip_flop: [], convert_bin_swap: []}

    nbits = range(10, 101, 10)
    for fitness_fn, records in records_by_prob.items():
        print(fitness_fn)
        for nbit in nbits:
            print(f"nbit={nbit}")
            records.extend(run_one(fitness_fn=fitness_fn, nbit=nbit, algos=algos))

    for prob, probn in zip([six_peaks, flip_flop, convert_bin_swap],
                           ["six_peaks", "flip_flop", "convert_bin_swap"]):
        fig = plot_fitting_curve([rec for rec in records_by_prob[prob] if rec["nbit"] == 100])
        fig.savefig(f"..//output//iteration_{probn}_100bit.png")
        fig = plot_vs_nbit([rec for rec in records_by_prob[prob]])
        fig.savefig(f"..//output//vs_bit_{probn}.png")
