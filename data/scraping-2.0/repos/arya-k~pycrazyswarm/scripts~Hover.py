"""
Uses StableBaselines package, forked from OpenAI baselines,
to meet the target env HoverEnv's goal. Is able to hover
a drone at a height 1m above the origin. Uses varied training
by not always starting the drone at the same spot. This helps address
potential instabilities in flight paths.
"""

import gym
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from crazyenv.env import HoverEnv

env = DummyVecEnv([lambda: HoverEnv()])

policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[200, 200])

model = PPO2(
    MlpPolicy, env,
    gamma=0.99,
    n_steps=256,
    policy_kwargs=policy_kwargs,
    nminibatches=256,
    learning_rate=5e-4,
    cliprange=0.2,
    tensorboard_log='/home/um/tensorboards/',
    verbose=1
)

# iterated training, in case the final results are unstable.
for i in range(100):
    model.learn(
        total_timesteps=300000,
        log_interval=1
    )
    print("Finished training {} times.".format(i))
    model.save("Hover1")
