#!/usr/bin/env python3

# import rospy
# import rospkg

import gym
import numpy as np
# from openai_ros.task_envs.deepleng import deepleng_docking
from stable_baselines.bench import Monitor
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.callbacks import BaseCallback

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor


if __name__ == "__main__":
    class InfoCallback(BaseCallback):
        """
        Callback for saving a model (the check is done every ``check_freq`` steps)
        based on the training reward (in practice, we recommend using ``EvalCallback``).

        :param check_freq: (int)
        :param log_dir: (str) Path to the folder where the model will be saved.
          It must contains the file created by the ``Monitor`` wrapper.
        :param verbose: (int)
        """

        def __init__(self, check_freq: int, verbose=1):
            super(InfoCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.best_mean_reward = -np.inf

        def _on_training_start(self) -> None:
            """
            This method is called before the first rollout starts.
            """
            print("Started training")
            # print("parameters: ", self.model.get_parameters())


        def _on_step(self) -> bool:
            print("num timesteps: ", self.num_timesteps)
            # print("observation: ", self.model.mb_obs)
            # print("Rewards: ", self.model.rewards)
            # print("Info: ", self.infos)
            # print("actions: ", self.actions)
            return True
    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('deepleng_control')
    # outdir = pkg_path + '/training_results/'

    # rospy.init_node('stable_baselines_lander', anonymous=True)
    # rospy.init_node('stable_baselines_docker', anonymous=True)
    env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('DeeplengDocking-v1')
    # check_env(env)
    model = PPO2(MlpPolicy,
                 env,
                 n_steps=1024,
                 nminibatches=32,
                 verbose=1,
                 lam=0.98,
                 gamma=0.999,
                 noptepochs=4,
                 ent_coef=0.01,
                 # tensorboard_log="/home/dfki.uni-bremen.de/mpatil/Documents/baselines_log",
                 seed=1)

    model.learn(total_timesteps=int(1e5), log_interval=50)
    # model.learn(total_timesteps=int(1e5), log_interval=50, tb_log_name="ppo_Lander_1e5")

    # model.save("/home/dfki.uni-bremen.de/mpatil/Desktop/ppo_LunarLander")
    # del model

    # model = PPO2.load("/home/dfki.uni-bremen.de/mpatil/Desktop/ppo_LunarLander")

    # print("Enjoy the trained agent")
    # obs = env.reset()
    # for i in range(10000):
    #     action, _states = model.predict(obs)
    #     # print("action:", action)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
    #     if dones:
    #         obs = env.reset()
    print("Closing environment")
    env.close()