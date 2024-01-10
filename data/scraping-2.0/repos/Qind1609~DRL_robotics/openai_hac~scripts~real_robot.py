#! /usr/bin/env python3

import random
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import rospy
import torch
import gym
import os
from her.ddpg_her_normalization import *
from her.normalizer import normalizer
import numpy as np
from mpi4py import MPI
import time
from openai_ros.task_envs.nachi import nachi_random_world
from openai_ros.robot_envs import nachi_env
from openai_ros import robot_gazebo_env
from Real_robot.Nachi_Comm import Socket_comm

class Hyper_Params:
    def __init__(self):
        self.env_name = "NachiReach_v0"
        self.seed = 1506

        # number of epochs for training
        self.num_epochs = 100

        # number of episodes - the times to collect samplers per epoch (reset -> new goal ->action)
        self.num_episodes = 50

        # maximum step for 1 episode
        self.max_ep_step = 300  # steps

        # the times to update networks
        self.num_batches = 50  # (divide the dataset into 50 batch, shuffle and random sample for each batch)

        # batch size
        self.batch_size = 300

        # initial number of step for random exploration
        # self.start_steps = 10000

        # size of replay buffer
        self.buff_size = 1000000  #  buffer size => 1000000 transitions

        # test phase
        self.phase = "test"

        # path to save model
        self.save_dir = (
            "/home/qind/Desktop/catkin_ws/src/openai_hac/scripts/her/saved_models"
        )

        # number of episodes testing should run
        self.test_episodes = 100

        # the clip ratio
        self.clip_obs = np.inf

        # the clip range
        self.clip_range = np.inf

        # learning rate actor
        self.lr_actor = 0.001

        # learning rate critic
        self.lr_critic = 0.001

        # scaling factor for gausian noise on action
        self.noise_eps = 0.1

        # random epsilon
        self.random_eps = 0.3

        # discount factor in bellman equation
        self.gamma = 0.98

        # polyak value for averaging
        self.polyak = 0.95

        # cuda - using GPU?
        self.cuda = True

        # number of worker (load data from cpu to gpu)
        # self.num_workers = 1

        # the rollout per MPI
        self.num_rollouts_per_mpi = 2

        # l2 regularization
        self.action_l2 = 1

        # replay_k
        self.replay_k = 4

        # threshold success
        self.threshold = 0.005

        # training space
        self.position_x_max = 0.63
        self.position_x_min = 0.3
        self.position_y_max = 0.145
        self.position_y_min = -0.145
        self.position_z_max = 0.31
        self.position_z_min = 0.15

class Real_Move():
    def __init__(self, params, env, env_params, robot) -> None:
        self.env = env
        self.env_params = env_params
        self.params = params
        self.robot = robot
        # load actor model
        self.actor = Actor(self.env_params)
        
        # create the normalizer
        self.o_norm = normalizer(
                size=self.env_params["obs_dim"], default_clip_range=self.params.clip_range
            )
        self.g_norm = normalizer(
                size=self.env_params["goal_dim"], default_clip_range=self.params.clip_range
            )   
        if MPI.COMM_WORLD.Get_rank() == 0:
            if os.path.exists(
                os.path.join(
                    self.params.save_dir, self.params.env_name, "actor_critic.pt"
                )
            ):
                checkpoint = torch.load(
                    os.path.join(
                        self.params.save_dir, self.params.env_name, "actor_critic.pt"
                    )
                )
                self.actor.load_state_dict(checkpoint["actor_state_dict"])

                self.actor.eval()
                self.g_norm.mean = checkpoint['g_mean']
                self.o_norm.mean = checkpoint['o_mean']
                self.g_norm.std = checkpoint['g_std']
                self.o_norm.std = checkpoint['o_std']
        
        if self.params.cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.actor.to(device)

    def move_HER_IK(self, x, y, z):

        # set goal data (input)
        g = np.array([x, y, z])
        self.env.set_goal(x, y, z, 0, 0, 0, 1.0)
        obs = self.env.get_obs()
        print(obs["desired_goal"])
        o = obs['observation']
        g = obs['desired_goal']
        Done = False
        step = 0
        while not (Done or (step == self.params.max_ep_step)):
                with torch.no_grad():

                    # pre-process input
                    input_tensor = self._preproc_inputs(o, g)

                    #feed to actor
                    pi = self.actor(input_tensor)

                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()

                # set action in virtual env
                observation_new, _, Done, _ = self.env.step(actions)
                current_pose = self.robot.tool_coordinate()
                action = list(np.array(current_pose[:3]) + np.array(actions[:3]))
                actions = np.concatenate([action, current_pose[3:]]).tolist()
                print(actions)
                # set action for real-robot here
                self.robot.moveposition(actions, 'machine_abs_linear')
                # get new observation
                o = observation_new["observation"]
                step += 1
        
        rospy.logwarn(
            "####################### Complete Move ##########################"
        )
        print(">>>>>> Complete in {0} steps <<<<<<<".format(step))

    def _preproc_inputs(self,o, g):
        obs_norm = self.o_norm.normalize(o)
        g_norm = self.g_norm.normalize(g)

        inputs = np.concatenate([obs_norm, g_norm])

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if params.cuda:
            inputs = inputs.cuda()
        return inputs

if __name__ == "__main__":
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["IN_MPI"] = "1"

    params = Hyper_Params()
    rospy.init_node("HER_reach")
    
    # init virtual env
    task_and_robot_environment_name = rospy.get_param(
        "/Nachi/task_and_robot_environment_name"
    )

    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name, params.max_ep_step
    )

    # set seed - make sure model give same result every run
    env.seed(params.seed + MPI.COMM_WORLD.Get_rank())
    env.action_space.seed(params.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(params.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(params.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(params.seed + MPI.COMM_WORLD.Get_rank())
    
    if params.cuda:
        torch.cuda.manual_seed(params.seed + MPI.COMM_WORLD.Get_rank())
    
    # connect to real robot TODO:
    Robot = Socket_comm()
    Robot.socket_initalize()

    #Initialize robot
    # reset and get observation (vir robot)
    obs = env.reset()

    goal = obs["desired_goal"]
    # move home real robot TODO:
    Robot.move_home()

    print("Home Position Joint {0}".format(Robot.joint_coordinate)+"/n"+"Position Tool {0}".format(Robot.tool_coordinate()))

    print("initial random goal is: {0}".format(goal))
    env_params = {
        "obs_dim": obs["observation"].shape[0],
        "goal_dim": obs["desired_goal"].shape[0],
        "action_dim": env.action_space.shape[0],
        "action_max": env.action_space.high[0],
        "max_timesteps": env._max_episode_steps,  # max_step for each ep
    }

    ddpg_agent = Real_Move(params, env, env_params, Robot)
    
    print(">>>>>>>>>>>>>>>>>>> Input your goal: ..... <<<<<<<<<<<<<")
    while 1:
        x = float(input('Input X goal (m): '))
        y = float(input('Input Y goal (m): '))
        z = float(input('Input Z goal (m): '))    
        ddpg_agent.move_HER_IK(x,y,z)            # coordinate
        
        time.sleep(4)
