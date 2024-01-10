# The structure of this model is adapted from the reference implementation of CartPole
# documented at https://tensorforce.readthedocs.io/en/latest/. We made 

import numpy as np
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import matplotlib.pyplot as plt
import sys

env = OpenAIGym('CartPole-v0', visualize=False)

training_progress = []

agent = PPOAgent(
    states=env.states,
    actions=env.actions,
    network=[
        dict(type='dense', size=32, activation='relu'),
        dict(type='dense', size=32, activation='relu'),
        dict(type='dense', size=32, activation='relu'),
        dict(type='dense', size=32, activation='relu'),
    ],
    batching_capacity=4096,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    optimization_steps=10,
    scope='ppo',
    discount=0.99,
    entropy_regularization=0.01,
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    likelihood_ratio_clipping=0.2,
)

if "--resume" in sys.argv:
    agent.restore_model(directory="models/")

runner = Runner(agent=agent, environment=env)

def episode_finished(r):
    print("[{ep}] @ {ts}ts -> \t{reward}".format(ep=r.episode, ts=r.episode_timestep,
                                                 reward=r.episode_rewards[-1]))
    training_progress.append(r.episode_rewards[-1])
    if r.episode % 100 == 0:
        env.visualize = True
        agent.save_model(directory="models/")
        plt.scatter(range(len(training_progress)), training_progress, s=1)
        plt.title("Cart Pole Training Progress\n3-layer 10-neurons/layer ReLU")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.savefig(fname="training_progress.png")
    else:
        env.visualize = False
    return True

runner.run(max_episode_timesteps=200, episodes=5000, episode_finished=episode_finished)
runner.close()