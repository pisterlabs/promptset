import numpy as np
import gym
from gym import wrappers
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from openai import OpenAI

class VIPIExperiment():
    def __init__(self, problem="Taxi-v3", run_q=False):
        if type(problem)==str:
            self.mdp = OpenAI(problem)
            self.P, self.R = self.mdp.P, self.mdp.R
        else:
            self.mdp = problem
            self.P, self.R = self.mdp
        self.df_vi = pd.DataFrame()
        self.df_pi = pd.DataFrame()
        self.df_q = pd.DataFrame()
        
        self.run_vi()
        self.run_pi()

        if run_q:
            self.run_q()
    
    def run_vi(self):
        for gamma in [0.1, 0.5, 0.9]:
            temp_df = pd.DataFrame()
            vi = ValueIteration(self.P, self.R, max_iter=int(1e10), gamma=gamma)
            temp_df = pd.DataFrame(vi.run())
            temp_df["gamma"] = gamma
            self.df_vi = self.df_vi.append(temp_df)
    
    def run_pi(self):
        for gamma in [0.1, 0.5, 0.9]:
            temp_df = pd.DataFrame()
            vi = PolicyIteration(self.P, self.R, max_iter=int(1e10), gamma=gamma)
            temp_df = pd.DataFrame(vi.run())
            temp_df["gamma"] = gamma
            self.df_pi = self.df_pi.append(temp_df)
    
    def run_q(self):
        for eps in [0, 0.5, 1]:
            temp_df = pd.DataFrame()
            for alpha in [0.1, 0.5, 1]:
                vi = QLearning(self.P, self.R, gamma=0.9, epsilon=eps, alpha=alpha, n_iter=100000)
                temp_df = pd.DataFrame(vi.run())
                temp_df["epsilon_start"] = eps
                temp_df["alpha_start"] = alpha
                self.df_q = self.df_q.append(temp_df)
    
    def gamma_plot(self, df, metric="Mean V"):
        plt.plot(df[df.gamma==0.1][metric], label="gamma=0.1")
        plt.plot(df[df.gamma==0.5][metric], label="gamma=0.5")
        plt.plot(df[df.gamma==0.9][metric], label="gamma=0.9")
        plt.legend()
   
    def comparative_plot(self, metric="Mean V"):
        plt.plot(self.df_pi[metric], label="Policy Iteration")
        plt.plot(self.df_vi[metric], label="Value Iteration")
        plt.plot(self.df_q[metric], label="Q Learning")
        plt.legend()
   
    def plot_eps(self, df, metric="Mean V"):
        plt.plot(df[df.epsilon_start==0][metric], label="epsilon=0")
        plt.plot(df[df.epsilon_start==0.5][metric], label="epsilon=0.5")
        plt.plot(df[df.epsilon_start==1][metric], label="epsilon=1")
        plt.legend()
   
    def plot_alpha(self, df, metric="Mean V"):
        plt.plot(df[df.alpha_start==0.1][metric], label="alpha=0.1")
        plt.plot(df[df.alpha_start==0.5][metric], label="alpha=0.5")
        plt.plot(df[df.alpha_start==1][metric], label="alpha=1")
        plt.legend()


        
        