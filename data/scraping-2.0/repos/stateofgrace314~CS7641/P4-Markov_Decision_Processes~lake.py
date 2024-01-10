"""
Function for running the frozen lake problem and analysis
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import OUTDIR, RunVI, RunPI, RunQ, plot_QL

DIRS = {0: "←", 1: "↓", 2: "→", 3: "↑"}

def plot_lake(env, policy=None, model=""):
    """
    Plot the Optimal policy (map) for the Frozen Lake
    """
    # convert env desc to plot colors
    colors = { b"S": "b", b"F": "w", b"H": "k", b"G": "g"}
    sqs = env.nrow
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, xlim=(-.01, sqs+0.01), ylim=(-.01, sqs+0.01))
    title = f"Frozen Lake Policy - {model}"
    plt.title(title, fontsize=12, weight="bold")
    for i in range(sqs):
        for j in range(sqs):
            y = sqs - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1, linewidth=1, edgecolor="k")
            p.set_facecolor(colors[env.desc[i,j]])
            ax.add_patch(p)
            if policy is not None:
                ax.text(x+0.5, y+0.5, DIRS[policy[i, j]], horizontalalignment="center", size=25,
                        verticalalignment="center", color="k")
    plt.axis("off")
    title = f"FrozenLakeOptimalPolicy{model}"
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{title}.png", dpi=400)

# code based on:
# https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438
def get_score(env, policy, verbose=False, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()[0]
        steps=0
        while True:
            action = policy.flatten()[observation]
            observation, reward, done, *_ = env.step(action)
            steps+=1
            if done and reward == 1:
                steps_list.append(steps)
                break
            elif done and reward == 0:
                misses += 1
                break
    ave_steps = np.mean(steps_list)
    std_steps = np.std(steps_list)
    pct_fail  = (misses/episodes)* 100

    if verbose:
        print("----------------------------------------------")
        print(f"You took an average of {ave_steps:.0f} steps to get the frisbee")
        print(f"And you fell in the hole {pct_fail:.2f} % of the times")
        print("----------------------------------------------")

    return ave_steps, std_steps, pct_fail

def getStats(data, env, verbose=True):
    """
    Adds some additional stats that weren"t needed for the frozen lake problem
    """
    # add a few columns
    data["average_steps"] = [0]*len(data)
    data["steps_stddev"] = [0]*len(data)
    data["success_pct"] = [0]*len(data)
    policies = data["policy"]
    for i in range(len(policies)):
        data["policy"][i] = np.array(data["policy"][i]).reshape(4,4)
        policy = data["policy"][i]
        steps, steps_stddev, failures = get_score(env, policy, verbose=False)
        data["average_steps"][i] = steps
        data["steps_stddev"][i] = steps_stddev
        data["success_pct"][i] = 100-failures

def plotAvg(data, x, independent, dependent, title, model):
    """
    Plot helper function
    """
    y = [data.loc[data[independent] == xi][dependent].mean() for xi in x]
    sns.set(style="whitegrid")

    plt.figure(figsize=(6,4))
    ax = sns.barplot(x=x, y=y)
    ax.set_title(title)
    ax.set_xlabel(independent)
    ax.set_ylabel(dependent)

    title=f"FrozenLake{independent}vs{dependent}{model}"
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{title}.png", dpi=400)

def runFrozenLake(verbose=True):
    """
    Runs the RL analysis on the frozen lake problem
    """
    # Setup 4x4 frozen lake
    env = gym.make("FrozenLake-v1").unwrapped
    env.max_episode_steps=500

    # Create transition and reward matrices from OpenAI P matrix
    rows = env.nrow
    cols = env.ncol
    T = np.zeros((4, rows*cols, rows*cols))
    R = np.zeros((4, rows*cols, rows*cols))

    # Loop through the env to build the lake
    state0 = np.inf
    # squares in the map
    for sq in env.P:
        # actions allowed
        for a in env.P[sq]:
            # resulting states
            for i in range(len(env.P[sq][a])):
                state1 = env.P[sq][a][i][1]
                if state1 == state0:
                    T[a][sq][state1] = T[a][sq][state0] + env.P[sq][a][i][0]
                    R[a][sq][state1] = R[a][sq][state0] + env.P[sq][a][i][2]
                else:
                    T[a][sq][state1] = env.P[sq][a][i][0]
                    R[a][sq][state1] = env.P[sq][a][i][2]
                state0 = state1
    # print out the default (empty) lake
    plot_lake(env, model="Default")

    ### Value Iteration ###

    # Parameters to iterate
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    epsilons = [1e-1, 1e-2, 1e-3, 1e-5, 1e-7, 1e-10, 1e-12]

    # run the analysis
    vi_data = RunVI(T, R, gammas, epsilons, verbose=verbose)
    getStats(vi_data, env, verbose)
    vi_data.to_csv(f"{OUTDIR}/FrozenLakeDataVI.csv")

    # plot reward vs gamma
    vi_data.plot(x="gamma", y="reward", title="Frozen Lake Reward vs. Gamma")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/FrozenLakeRewardvsGammaVI.png")
    # plot reward vs iterations
    vi_data.sort_values("iterations", inplace=True)
    vi_data.plot(x="iterations", y="reward", title="Frozen Lake Reward vs. Iterations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/FrozenLakeRewardvsIterationVI.png")
    # plot success pct vs gamma
    vi_data.plot(x="gamma", y="success_pct", title="Frozen Lake Success Rate vs. Gamma")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/FrozenLakeSuccessvsGammaVI.png")
    # plot avg steps vs gamma
    plotAvg(vi_data, gammas, "gamma", "average_steps", "Avg Steps vs Gamma", "VI")
    plotAvg(vi_data, gammas, "gamma", "success_pct", "Success Rate vs Gamma", "VI")

    # get the best one, plot it and save the data
    bestRun = vi_data["reward"].argmax()
    bestR = vi_data["reward"].max()
    bestPolicy = vi_data["policy"][bestRun]
    bestG = vi_data["gamma"][bestRun]
    bestE = vi_data["epsilon"][bestRun]
    bestPct = vi_data["success_pct"][bestRun]
    plot_lake(env, bestPolicy, "VI")
    if verbose:
        print(f"Best Result:\n\tReward = {bestR:.2f}\n\tGamma = {bestG:.3f}\n\tEps= {bestE:.2E}\n\tSuccess Rate = {bestPct:.2f}")

    ### Policy Iteration ###

    # Parameters to iterate
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]

    # run the analysis
    pi_data = RunPI(T, R, gammas, verbose=verbose)
    getStats(pi_data, env, verbose)
    pi_data.to_csv(f"{OUTDIR}/FrozenLakeDataPI.csv")

    # plot reward vs gamma
    pi_data.plot(x="gamma", y="reward", title="Frozen Lake Reward vs. Gamma")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/FrozenLakeRewardvsGammaPI.png")
    # plot reward vs iterations
    pi_data.sort_values("iterations", inplace=True)
    pi_data.plot(x="iterations", y="reward", title="Frozen Lake Reward vs. Iterations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/FrozenLakeRewardvsIterationPI.png")
    # plot success pct vs gamma
    pi_data.plot(x="gamma", y="success_pct", title="Frozen Lake Success Rate vs. Gamma")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/FrozenLakeSuccessvsGammaPI.png")
    # plot avg steps vs gamma
    plotAvg(pi_data, gammas, "gamma", "average_steps", "Avg Steps vs Gamma", "PI")
    plotAvg(pi_data, gammas, "gamma", "success_pct", "Success Rate vs Gamma", "PI")

    # get the best one, plot it and save the data
    bestRun = pi_data["reward"].argmax()
    bestR = pi_data["reward"].max()
    bestPolicy = pi_data["policy"][bestRun]
    bestG = pi_data["gamma"][bestRun]
    bestPct = pi_data["success_pct"][bestRun]
    plot_lake(env, bestPolicy, "PI")
    if verbose:
        print(f"Best Result:\n\tReward = {bestR:.2f}\n\tGamma = {bestG:.3f}\n\tSuccess Rate = {bestPct:.2f}")

    ### Q learning ###
    gammas         = [0.8, 0.9, 0.99]
    alphas         = [0.01, 0.1, 0.2]
    alpha_decays   = [0.9, 0.999]
    epsilon_decays = [0.9, 0.999]
    iterations     = [1e4, 1e5, 1e6]

    # # to run again, uncommment below (takes a long time)
    # ql_data  = RunQ(T, R, gammas, alphas, alpha_decays=alpha_decays, epsilon_decays=epsilon_decays,
    #                 n_iterations=iterations, verbose=verbose)
    # getStats(ql_data, env, verbose)
    # file = f"{OUTDIR}/FrozenLakeDataQl.csv"
    # ql_data.to_csv(file)

    # Read in Q-Learning data
    ql_data = pd.read_csv(f"{OUTDIR}/FrozenLakeDataQl.csv")

    # check which hyperparameters made most impact
    slice = ["gamma", "alpha", "alpha_decay", "epsilon_decay", "iterations", "reward", "time"]
    df = ql_data[slice]
    ql_corr = df.corr()
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(8,7))
    ax.set_title("Correlation Matrix of Q-Learning Parameters", fontsize=20)
    mask = np.triu(np.ones_like(ql_corr, dtype=np.bool))
    cmap = sns.diverging_palette(255, 0, as_cmap=True)
    sns.heatmap(ql_corr, mask=mask, cmap=cmap, square=True, linewidths=0.5, cbar_kws={"shrink":.75})
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/FrozenLakeQlParamCorrelation.png")

    # Plots
    plot_QL(ql_data, "iterations", "time", "FrozenLake", title="Mean Runtime vs. Num Iterations", logscale=True)
    plot_QL(ql_data, "iterations", "success_pct", "FrozenLake", title="Success Rate vs. Num Iterations", logscale=True)
    plot_QL(ql_data, "alpha_decay", "success_pct", "FrozenLake", title="Success Rate vs. Alpha Decay")
    plot_QL(ql_data, "gamma", "success_pct", "FrozenLake", title="Success Rate vs. Gamma")
    plot_QL(ql_data, "gamma", "reward", "FrozenLake", title="Reward vs. Gamma")
    plotAvg(ql_data, gammas, "gamma", "average_steps", "Avg Steps vs Gamma", "Ql")
    plotAvg(ql_data, gammas, "gamma", "success_pct", "Success Rate vs Gamma", "Ql")

    # Get the best one
    bestRun = ql_data["reward"].argmax()
    best_policy = ql_data["policy"][bestRun]
    bestR = ql_data["reward"][bestRun]
    bestG = ql_data["gamma"][bestRun]
    bestA = ql_data["alpha"][bestRun]
    bestAD = ql_data["alpha_decay"][bestRun]
    bestED = ql_data["epsilon_decay"][bestRun]
    bestIt = ql_data["iterations"][bestRun]
    bestPct = ql_data["success_pct"][bestRun]

    # plot the policy (convert if necessary)
    if isinstance(best_policy, str):
        best_policy = best_policy.split("\n")
        best_policy = [row.strip("[").strip().strip("]").strip("]").strip("[").split() for row in best_policy]
        best_policy = np.array(best_policy).astype(int)
    plot_lake(env, best_policy, "Ql")

    if verbose:
        print(f"Best Result:\n\tReward = {bestR:.2f}\n\tGamma = {bestG:.3f}\n\tAlpha = {bestA:.3f}"
              f"\n\tAlpha Decay = {bestAD:.3f}\n\tEps Decay = {bestED:.3f}\n\tIterations = {bestIt}"
              f"\n\tSuccess Rate = {bestPct:.2f}")
    bestPct = ql_data["success_pct"].max()
    print(f"\tBest Success Rate = {bestPct:.2f}")