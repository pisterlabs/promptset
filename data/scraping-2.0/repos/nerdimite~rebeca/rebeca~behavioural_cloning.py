# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.

from argparse import ArgumentParser
import pickle
import time
from tqdm.auto import tqdm

import gym
import minerl
import torch as th
import numpy as np
import wandb
from transformers import get_linear_schedule_with_warmup

from openai_vpt.agent import PI_HEAD_KWARGS, MineRLAgent
from data_loader import DataLoader
from openai_vpt.lib.tree_util import tree_map

# Originally this code was designed for a small dataset of ~20 demonstrations per task.
# The settings might not be the best for the full BASALT dataset (thousands of demonstrations).
# Use this flag to switch between the two settings
USING_FULL_DATASET = True

EPOCHS = 1 if USING_FULL_DATASET else 2
# Needs to be <= number of videos
BATCH_SIZE = 64 if USING_FULL_DATASET else 16
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 100 if USING_FULL_DATASET else 20
DEVICE = "cuda"

SAVE_FREQ = 100

# Tuned with bit of trial and error
LEARNING_RATE = 0.000181
# OpenAI VPT BC weight decay
# WEIGHT_DECAY = 0.039428
WEIGHT_DECAY = 0.0
# KL loss to the original model was not used in OpenAI VPT
KL_LOSS_WEIGHT = 1.0
MAX_GRAD_NORM = 5.0

MAX_BATCHES = 20000 if USING_FULL_DATASET else int(1e9)

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def behavioural_cloning_train(data_dir, in_model, in_weights, save_dir):

    wandb.init(
        project="rebeca-mp",
        entity="nerdimite",
        config={
            "model_config": in_model,
        },
        tags=["dummy_run"]
    )

    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    agent = MineRLAgent(device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)

    policy = agent.policy

    # Freeze all parameters first
    for param in policy.parameters():
        param.requires_grad = False

    # Unfreeze recurrent layers
    trainable_parameters = []
    for param in policy.net.recurrent_layer.blocks[-2].parameters():
        param.requires_grad = True
        trainable_parameters.append(param)
    for param in policy.net.recurrent_layer.blocks[-1].parameters():
        param.requires_grad = True
        trainable_parameters.append(param)
    # Unfreeze final layers
    for param in policy.net.lastlayer.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)
    for param in policy.pi_head.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)

    # Parameters taken from the OpenAI VPT paper
    optimizer = th.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=MAX_BATCHES
    )

    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
    )

    # Keep track of the hidden state per episode/trajectory.
    # DataLoader provides unique id for each episode, which will
    # be different even for the same trajectory when it is loaded
    # up again
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

    pbar = tqdm(total=MAX_BATCHES)

    loss_sum = 0
    for batch_i, (batch_images, batch_actions, batch_episode_id) in enumerate(data_loader):
        batch_loss = 0
        for image, action, episode_id in zip(batch_images, batch_actions, batch_episode_id):
            if image is None and action is None:
                # A work-item was done. Remove hidden state
                if episode_id in episode_hidden_states:
                    removed_hidden_state = episode_hidden_states.pop(episode_id)
                    del removed_hidden_state
                continue

            agent_action = agent._env_action_to_agent(action, to_torch=True, check_if_null=True)
            if agent_action is None:
                # Action was null
                continue

            agent_obs = agent._env_obs_to_agent({"pov": image})
            if episode_id not in episode_hidden_states:
                episode_hidden_states[episode_id] = policy.initial_state(1)
            agent_state = episode_hidden_states[episode_id]

            pi_distribution, _, new_agent_state = policy.get_output_for_observation(
                agent_obs,
                agent_state,
                dummy_first
            )

            log_prob  = policy.get_logprob_of_action(pi_distribution, agent_action)

            # Make sure we do not try to backprop through sequence
            # (fails with current accumulation)
            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            episode_hidden_states[episode_id] = new_agent_state

            # Finally, update the agent to increase the probability of the
            # taken action.
            # Remember to take mean over batch losses
            loss = -log_prob / BATCH_SIZE
            batch_loss += loss.item()
            loss.backward()

        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        pbar.set_postfix({"batch_loss": batch_loss / BATCH_SIZE, "lr": scheduler.get_last_lr()})
        pbar.update(1)
        wandb.log({"batch_loss": batch_loss / BATCH_SIZE, "lr": scheduler.get_last_lr()})

        if batch_i % SAVE_FREQ == 0 and batch_i > 0:
            state_dict = agent.controller.state_dict()
            th.save(state_dict, f"{save_dir}/{wandb.run.name}_{batch_i}.weights")

        if batch_i > MAX_BATCHES:
            break

    state_dict = policy.state_dict()
    th.save(state_dict, f"{save_dir}/{wandb.run.name}.weights")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/MakeWaterfallTrain/", help="Path to the directory containing recordings to be trained on")
    parser.add_argument("--in-model", default="data/VPT-models/foundation-model-1x.model", type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", default="data/VPT-models/foundation-model-1x.weights", type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--save-dir", default="models", type=str, help="Path where finetuned weights will be saved")

    args = parser.parse_args()
    behavioural_cloning_train(args.data_dir, args.in_model, args.in_weights, args.save_dir)
