""" Loads saved PPO policies that can be used for evaluation.  Separate
evaluation from the main Elo Rating framework used to evaluate Dreamer and PPO
agents. """

# Native Python imports
import os
from datetime import datetime
import argparse

# TensorFlow and tf-agents
from tf_agents.environments import gym_wrapper, tf_py_environment
import tf_agents.trajectories.time_step as ts

# Other external packages
import numpy as np
import tensorflow as tf

# Custom packages
from utils import video_summary, encode_gif, ObservationWrapper
from envs.multi_car_racing import MultiCarRacing


def load_saved_policies(eval_model_path=None, train_model_path=None):
    """ Loads saved PPO policies from the local file system.

    Loads saved policies for an agent, depending on what paths are
    specified in the function call.  This method sets the agent's
    policies to be the new policies that are loaded below.

    Arguments:
        eval_model_path (str): A file path (relative or absolute) to the
            directory containing the policy that will be loaded as the
            evaluation policy of the agent.
        train_model_path (str): A file path (relative or absolute) to the
            directory containing the policy that will be loaded as the
            training policy of the agent.

    Returns:
        collect_policy (TF Agents Policy):  A tf-agents policy object
            corresponding to the training/exploratory policy of the PPO agent.
        eval_policy (TF Agents Policy):  A tf-agents policy object
            corresponding to the evaluation/greedy policy of the PPO agent.
    """
    # Load evaluation and/or training policies from path
    if eval_model_path is not None:
        eval_policy = tf.saved_model.load(eval_model_path)
        print("Loading evaluation policy from: {}".format(eval_model_path))

    if train_model_path is not None:
        collect_policy = tf.saved_model.load(train_model_path)
        print("Loading training policy from: {}".format(train_model_path))

    return collect_policy, eval_policy


def make_env():
    """ Function for creating the TfPyEnvironment from OpenAI Gym environment.
    """
    # Create wrapped environment
    gym_eval_env = MultiCarRacing()
    gym_eval_env.observation_space.dtype = np.float32  # For Conv2D data input

    # Now create Python environment from gym env
    py_eval_env = gym_wrapper.GymWrapper(gym_eval_env)  # Gym --> Py

    # Create training and evaluation TensorFlow environments
    tf_eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)  # Py --> Tf

    # Display environment specs
    print("Observation spec: {} \n".format(tf_eval_env.observation_spec()))
    print("Action spec: {} \n".format(tf_eval_env.action_spec()))
    print("Time step spec: {} \n".format(tf_eval_env.time_step_spec()))

    return tf_eval_env


def parse_args():
    """Command-line argument parser."""
    # Create parser object
    parser = argparse.ArgumentParser()

    # Add path arguments
    parser.add_argument("-p", "--path_to_policies", type=str, default="./",
                         help="Path to policies for evaluation.")
    parser.add_argument("-exp_name", "--experiment_name", type=str,
                        default="Evaluation episode",
                        help="The name of the evaluation experiment.")
    return parser.parse_args()


def main():
    """ Main function for running evaluation.
    """
    # Parse arguments
    args = parse_args()

    # Get paths for policies
    path_to_policies = args.path_to_policies
    EVAL_MODEL_PATH = path_to_policies
    TRAIN_MODEL_PATH = path_to_policies
    EXPERIMENT_NAME = args.experiment_name
    LOG_DIR = os.path.join("exp_eval", EXPERIMENT_NAME)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    train_policy, eval_policy = load_saved_policies(train_model_path=TRAIN_MODEL_PATH,
                                                    eval_model_path=EVAL_MODEL_PATH)


if __name__ == "__main__":
    main()
