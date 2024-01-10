"""
Utility functions for running and playing back reinforcement learning algos.
This also supports running most algorithms from openAI baselines

Assumes you are using the modified version hosted here: https://github.com/sgillen/baselines which properly saves the
VecNormalize variables for mujoco environments.

The advantage of using this over the command line interface is that these functions automatically keep track of meta
data (what arguments were used for a run_util, how long did the run_util take, what were you trying to accomplish), and takes
care of loading a trained model just by specifying the name you saved it with

"""

import warnings

try:
    import baselines.run
except:
    warnings.warn("baselines install not found, only seagul loads will work", ImportWarning)

import gym
import dill
import subprocess
import time, datetime, json
import os
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_and_save_bs(arg_dict, run_name=None, description=None, base_path="/data/"):

    """
    Launches an openAI baselines algorithm and records the results

    If you don't pass run_name or description this function will call input, blocking execution

    This function is my entry point to the openAI baselines command line. I like to be able to programmatically specify
    the arguments to use, and I'd like to be able to keep track of other meta data, plus log everything to the same
    place. Furthermore I'd like to log my data in a format where I can just point to a save file and reconstruct the
    network structure needed to play the agent out in the environment or test it in a new one

    Args:
        arg_dict: dictionary of arguments, you need to use the exact name that openAI uses
        run_name: name to save the run_util under
        description: brief description of what you were trying to do with the run_util

    Returns:
        Does not return anything

    Example:

        from run_rl.run_baselines import run_baselines

        arg_dict = {
        'env' : 'su_cartpole-v0',
        'alg' : 'ppo2',
        'network' : 'mlp',
        'num_timesteps' : '2e4',
        'num_env' : '1'
        }

        run_baselines(arg_dict, run_name='test2', description='')

    """

    if run_name is None:
        run_name = input("please enter a name for this run: ")

    if description is None:
        description = input("please enter a brief description of the run: ")

    save_base_path = os.getcwd() + base_path
    save_dir = save_base_path + run_name + "/"
    save_path = save_dir + "saved_model"
    arg_dict["save_path"] = save_path

    baselines_path = baselines.run.__file__
    os.environ["OPENAI_LOGDIR"] = save_dir
    os.environ["OPENAI_LOG_FORMAT"] = "stdout,csv,tensorboard"

    argv_list = [baselines_path]  # first argument is the path of baselines.run_util

    for arg, value in arg_dict.items():
        argv_list.append("--" + str(arg) + "=" + value)

    start_time = time.time()
    baselines.run.main(argv_list)
    runtime = time.time() - start_time

    datetime_str = str(datetime.datetime.today())
    datetime_str = datetime_str.replace(" ", "_")
    runtime_str = str(datetime.timedelta(seconds=runtime))

    with open(save_dir + "info.json", "w") as outfile:
        json.dump(
            {
                "args": arg_dict,
                "metadata": {"date_time": datetime_str, "total runtime": runtime_str, "description": description},
            },
            outfile,
            indent=4,
        )


def run_sg(arg_dict, algo, run_name=None, run_desc=None, base_path="/data/", append_time=True):
    """
    Launches seaguls ppo2 and save the results without clutter

    If you don't pass run_name or description this function will call input, blocking execution

    Arguments:
        arg_dict: dictionary with arguments for algo
        algo: the algorithm from seagul.algos to use
        run_name: string for the name of the run, if None we will ask you for one
        run_desc: short description to save with the run, if None we will ask you for one, can pass an empty string
        base_path: directory where you want the runs stored
        append_time: bool, if true will append the current time to the run name
    """

    if run_name is None:
        run_name = input("please enter a name for this run: ")

    if run_desc is None:
        run_desc = input("please enter a brief description of the run: ")

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

    save_base_path = os.getcwd() + base_path
    save_dir = save_base_path + run_name

    if append_time:
        now = datetime.datetime.now()
        date_str = str(now.month) + "-" + str(now.day) + "_" + str(now.hour) + "-" + str(now.minute)
        save_dir = save_dir + "--" + date_str

    save_dir = save_dir + "/"

    start_time = time.time()
    t_model, rewards, var_dict = algo(**arg_dict)
    runtime = time.time() - start_time

    datetime_str = str(datetime.datetime.today())
    datetime_str = datetime_str.replace(" ", "_")
    runtime_str = str(datetime.timedelta(seconds=runtime))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    str_dict = {key: str(value) for key, value in arg_dict.items()}
    with open(save_dir + "info.json", "w") as outfile:
        json.dump(
            {
                "args": str_dict,
                "metadata": {
                    "date_time": datetime_str,
                    "total runtime": runtime_str,
                    "description": run_desc,
                    "git_sha": git_sha,
                },
            },
            outfile,
            indent=4,
        )

    with open(save_dir + "workspace", "wb") as outfile:
        del var_dict["env"]
        torch.save(var_dict, outfile, pickle_module=dill)

    with open(save_dir + "model", "wb") as outfile:
        torch.save(t_model, outfile, pickle_module=dill)
    
    print(f"saved run in {save_dir}, last reward was {rewards[-1]}")


def load_model(save_path, backend="baselines"):

    """
    Loads and plays back a trained model.

    You must either specify a relative directory with the ./notation, or the absolute path. 
    However absolute paths only work with mac or Linux.

    Parameters:
        save_path: a string with the name you want to load. You probably are running this file that looks like: ~/work/data/run1/run1
        to load it provide the string './data/run1'

        backend: a string, either 'baselines' or 'seagul' depending on which implementation you used

    Returns:
       returns the model and the environment

    Example:
        from run_rl.run_baselines import play_baselines
        model, env = play_baseline('./data/run1')
    """

    if save_path[-1] == "/":
        save_path = save_path[:-1]

    if save_path.split("/")[1] == "home" or save_path.split("/")[1] == "User":
        save_base_path = save_path
    else:
        save_base_path = os.getcwd() + save_path.split(".")[1]

    run_name = save_path.split("/")[-1]
    # load_dir = save_base_path + run_name + 'info.json'
    # arg_dict['load_path']

    if backend == "baselines":
        with open(save_base_path + "/" + "info.json", "r") as outfile:
            data = json.load(outfile)

        arg_dict = data["args"]
        arg_dict["num_timesteps"] = "0"
        arg_dict["num_env"] = "1"
        del arg_dict["save_path"]
        arg_dict["load_path"] = save_base_path + "/" + "saved_model"

        baselines_path = baselines.run.__file__
        argv_list = [baselines_path]  # first argument is the path of baselines.run_util

        for arg, value in arg_dict.items():
            argv_list.append("--" + str(arg) + "=" + value)

        #argv_list.append("--play")

        model = baselines.run.main(argv_list)

        return model

    elif backend == "seagul":
        with open(save_base_path + "/" + "info.json", "r") as infile:
            data = json.load(infile)  # , Loader=yaml.Loader)
            # arg_dict = data['arg_dict']

        with open(save_base_path + "/" + "model", "rb") as infile:
            model = torch.load(infile)

        return model

    else:
        raise ValueError("unrecognized backend: ", backend)


def load_workspace(save_path):
    if save_path[-1] == "/":
        save_path = save_path[:-1]

    if save_path.split("/")[1] == "home" or save_path.split("/")[1] == "User":
        save_base_path = save_path
    else:
        save_base_path = os.getcwd() + save_path.split(".")[1]

    with open(save_base_path + "/" + "workspace", "rb") as infile:
        workspace = torch.load(infile, pickle_module=dill)

    with open(save_base_path + "/" + "info.json", "r") as infile:
        data = json.load(infile)  # , Loader=yaml.Loader)

    with open(save_base_path + "/" + "model", "rb") as infile:
        model = torch.load(infile, pickle_module=dill)

    env_name = data["args"]["env_name"]
    env = gym.make(env_name)

    return model, env, data, workspace


if __name__ == "__main__":
    from seagul.rl.run_utils import run_sg
    from seagul.rl.ppo import PPOModel
    from seagul.nn import MLP
    from seagul.rl.ppo.ppo2 import ppo

    import torch
    import torch.nn as nn

    ## init policy, valuefn
    input_size = 4
    output_size = 1
    layer_size = 64
    num_layers = 3
    activation = nn.ReLU

    torch.set_default_dtype(torch.double)

    model = PPOModel(
        policy=MLP(input_size, output_size, num_layers, layer_size, activation),
        value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
        action_std=4,
    )

    arg_dict = {"env_name": "su_cartpole-v0", "model": model, "num_epochs": 10, "action_var_schedule": [10, 0]}

    run_sg(arg_dict, ppo)