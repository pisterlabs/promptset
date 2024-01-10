# -*- coding: utf-8 -*-
"""TD3 agent from demonstration for episodic tasks in OpenAI Gym.

- Author: Seungjae Ryan Lee
- Contact: seungjaeryanlee@gmail.com
- Paper: https://arxiv.org/pdf/1802.09477.pdf (TD3)
         https://arxiv.org/pdf/1511.05952.pdf (PER)
         https://arxiv.org/pdf/1707.08817.pdf (DDPGfD)
"""

import wandb
import numpy as np
import time
import torch
from torch.nn.utils import clip_grad_norm_
import pickle
import glob
import numpy as np
import torch
import os
import cv2
from collections import deque
from copy import deepcopy
import algorithms.common.helper_functions as common_utils
from algorithms.common.buffer.priortized_replay_buffer_np import PrioritizedReplayBufferfD
from algorithms.common.buffer.replay_buffer_np import NStepTransitionBuffer, ReplayBufferExplainer
from algorithms.sac.agent_mlp import Agent as SACAgent
from algorithms.fd.td3_cnn_agent import Agent as TD3FDAgent
from algorithms.fd.se_utils import img2simpleStates, simpleStates2img

# from openai_ros.task_envs.fetch_serl.utils.load_config_utils import loadYAML
from algorithms.common.load_config_utils import loadYAML
from algorithms.common.load_config_utils import loadYAML
from algorithms.common.helper_functions import draw_predicates_on_img
from kair_algorithms_draft.scripts.experiment_record_utils import ExperimentLogger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
make_data = True
class Agent(SACAgent):
    """SAC agent interacting with environment.

    Attrtibutes:
        memory (PrioritizedReplayBufferfD): replay memory
        beta (float): beta parameter for prioritized replay buffer

    """
    def __init__(self, env, args, hyper_params, models, optims, noises, her, robot_conf=None):
        SACAgent.__init__(self, env, args, hyper_params, models, optims, noises, her)
        self.hyper_params, self.robot_conf = hyper_params, robot_conf

        self.predicate_keys = self.robot_conf['env']['predicates_list']
        self.reached_goal_reward = self.robot_conf['env']['reached_goal_reward']
        self.explainer = models[5]
        self.explainer_optim = optims[4]
        self.reward_net = models[6]
        self.reward_net_optim = optims[5]

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        # return TD3Agent.__new__(self)

    # pylint: disable=attribute-defined-outside-init
    # def _initialize(self, env, args, hyper_params, models, optims, noises):
    def _initialize(self):
        """Initialize non-common things."""
        # conf_str, self.conf_data = loadYAML(os.getcwd() + "/../config/fetch_serl_push_env.yaml")
        conf_str, self.conf_data = loadYAML(self.args.robot_env_config)

        self.use_n_step = self.hyper_params["N_STEP"] > 1
        self.reached_goal_reward = self.conf_data['env']['reached_goal_reward']
        self.exe_single_group = self.conf_data['fetch']['exe_single_group']
        self.exe_group_num = self.conf_data['fetch']['exe_group_num']
        self.use_shaping = self.conf_data['env']['use_shaping']
        self.use_bi_reward = not self.use_shaping
        self.avg_scores_window = deque(maxlen=self.args.avg_score_window)
        self.total_steps = 0

        if not self.args.test:
            # load demo replay memory
            # TODO: should make new demo to set protocol 2
            #       e.g. pickle.dump(your_object, your_file, protocol=2)
            demo_files = glob.glob(self.args.demo_path + '/good*/traj*.pickle')
            demos = []
            self.demos = []
            for file in demo_files:
                with open(file, "rb") as f:
                    d = pickle.load(f, encoding="latin1")
                    demos.extend(d[0][self.hyper_params["DEMO_STARTS"]:])
                    self.demos.append(d[0])

            # Note the current environment is wrapped by "NormalizedActions", so here we need to normalize the demo actions to be within [-1, 1]
            # demos = common_utils.preprocess_demos(demos, resz=self.env.observation_space.shape[-1:-3:-1], action_space=self.env.action_space, reward_scale=50.0, exe_single_group=self.exe_single_group, exe_group_num=self.exe_group_num)
            demos = common_utils.preprocess_demos(demos, resz=self.env.observation_space.shape[-1:-3:-1], use_bi_reward=self.use_bi_reward, goal_reward=self.reached_goal_reward)

            if self.use_n_step:
                demos, demos_n_step = common_utils.get_n_step_info_from_demo(
                    demos, self.hyper_params["N_STEP"], self.hyper_params["GAMMA"]
                )

                # replay memory for multi-steps
                self.memory_n = NStepTransitionBuffer(
                    buffer_size=self.hyper_params["BUFFER_SIZE"],
                    n_step=self.hyper_params["N_STEP"],
                    gamma=self.hyper_params["GAMMA"],
                    demo=demos_n_step,
                )

            # replay memory
            self.beta = self.hyper_params["PER_BETA"]
            self.memory = PrioritizedReplayBufferfD(
                self.hyper_params["BUFFER_SIZE"],
                self.hyper_params["BATCH_SIZE"],
                demo=demos,
                alpha=self.hyper_params["PER_ALPHA"],
                epsilon_d=self.hyper_params["PER_EPS_DEMO"],
            )

            self.good_buffer = ReplayBufferExplainer(
                len(demos),
                batch_size=int(self.hyper_params["BATCH_SIZE"] / 2)
            )
            self.good_buffer.extend(transitions=demos)
            # sampled trajectories buffer (In Justin Fu's implementation, they don't reuse past sample)
            self.bad_buffer = ReplayBufferExplainer(
                self.hyper_params["BUFFER_SIZE"],
                batch_size=int(self.hyper_params["BATCH_SIZE"] / 2)
            )

    def _preprocess_state(self, state):
        """Preprocess state so that actor selects an action."""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        # if state is a single state, we unsqueeze it
        if len(state.size()) == 3:
            state = state.unsqueeze(0)
        return state

    def to_predicate_vector(self, predicate_value):
        """
        Convert dict representation of predicate values to list (vector) representation of predicate values
        :param predicate_value: a dict storing predicate values
        :return (np.ndarray): predicate vector
        """
        add_bias = self.hyper_params["BIAS_IN_PREDICATE"]
        predicate_vector = [0.0 for _ in predicate_value]
        if self.hyper_params["ONE_HOT_PREDICATES"]:
            predicate_vector = predicate_vector + predicate_vector
        # expand one 1 element for bias
        if add_bias:
            predicate_vector = predicate_vector + [1.0]
        for idx, pv in enumerate(predicate_value):
            if self.hyper_params["ONE_HOT_PREDICATES"]:
                if pv == 1:
                    predicate_vector[idx * 2 + 1] = 1
                else:
                    predicate_vector[idx * 2] = 1
            else:
                predicate_vector[idx] = pv
        return np.array(predicate_vector, dtype=np.float64)

    def to_predicate_vectors(self, predicate_values):
        return np.array([self.to_predicate_vector(predicate_value)
                         for predicate_value in predicate_values])

    # noinspection PyMethodMayBeStatic
    def _get_manual_shaping(self, state, predicate_values, next_state, next_predicate_values):
        """
        For comparison
        """

        # a = [["is_get_cube1", "is_cube1_insertedTo_block2"], ["is_get_cylinder1", "is_cylinder1_insertedTo_block1"]]
        # if len(a[0]) == len(a[1]) == 2:
        #     if curr_p == a[0][0]:
        #         a[0].pop()
        #     elif curr_p = a[1][0]:
        #         a[1].pop()
        # else:
        #     if

        shaping_rewards = np.zeros(shape=(len(predicate_values), 1))

        # procedure = deque(list_correct_predicates_changes)

        # if next_predicate_values == procedure.pop():
        #     shape_r = 1.0
            # How to deal with either cube1 or cyliner1 are equally fine with being grasped?



        for idx, predicate_value in enumerate(predicate_values):
            shaping_reward = 0
            next_predicate_value = next_predicate_values[idx]

            if not np.array_equal(predicate_value, next_predicate_value):
                shaping_rewards[idx][0] = 10.0
                continue
            # ofs = 0 if self.exe_single_group else 2
            ofs = 0
            sub_task_list = [["is_get_cube1", "is_cube1_insertedTo_block2"], ["is_get_cylinder1", "is_cylinder1_insertedTo_block1"]]
            simple_states = img2simpleStates(state, end=self.hyper_params["SIMPLE_STATES_SIZE"] - self.hyper_params[
                                                          "NUM_PREDICATES"])
            if 1.0 not in predicate_value:
                shaping_rewards[idx][0] = max(-np.linalg.norm(simple_states[idx][17-ofs:20-ofs]-simple_states[idx][23-ofs:26-ofs]), -np.linalg.norm(simple_states[idx][17-ofs:20-ofs]-simple_states[idx][26-ofs:29-ofs]))
            elif 1.0 not in predicate_value[2:]:
                for id, st in enumerate(sub_task_list):
                    if self.predicate_keys[np.where(predicate_value == 1.0)[0]] == st[0]:
                        if id == 0:
                            shaping_rewards[idx][0] = -np.linalg.norm(simple_states[idx][23-ofs:26-ofs]-simple_states[idx][35-ofs:38-ofs])
                        else:
                            shaping_rewards[idx][0] = -np.linalg.norm(simple_states[idx][26-ofs:29-ofs]-simple_states[idx][29-ofs:32-ofs])

            elif np.where(predicate_value == 1.0)[-1] == 2 or np.where(predicate_value == 1.0)[-1] == 4:
                shaping_rewards[idx][0] = -np.linalg.norm(simple_states[idx][17-ofs:20-ofs]-simple_states[idx][26-ofs:29-ofs])

            elif np.where(predicate_value == 1.0)[-1] == 3 or np.where(predicate_value == 1.0)[-1] == 5:
                shaping_rewards[idx][0] = -np.linalg.norm(
                    simple_states[idx][17 - ofs:20 - ofs] - simple_states[idx][23-ofs:26-ofs])

            print("SR", shaping_rewards[idx][0])
            # shaping_rewards[idx][0] = shaping_reward

        return shaping_rewards

    def get_shaping_reward(self, states, predicate_values, next_state, next_predicate_values):
        """
        Return shaping reward given the transition. The shaping reward is the biggest utility value change
            Note that shaping reward is given only when there is a predicate value change
        """
        if self.is_doing_pretrain or self.hyper_params["NO_SHAPING"]:
            return 0
        if self.hyper_params["MANUAL_SHAPING"]:
            return self._get_manual_shaping(states, predicate_values, next_state, next_predicate_values)

        predicate_vectors = self.to_predicate_vectors(predicate_values)
        states_util_values = self.get_states_utility(states, predicate_vectors)
        next_predicate_vectors = self.to_predicate_vectors(next_predicate_values)
        next_state_util_values = self.get_states_utility(next_state, next_predicate_vectors)

        shaping_rewards = next_state_util_values - states_util_values

        if self.hyper_params["PRINT_SHAPING"]:
            for idx, _ in enumerate(shaping_rewards):
                shaping_reward = shaping_rewards[idx][0]
                if shaping_reward != 0:
                    print('\n[Explainer INFO] shaping reward: ', shaping_reward,
                          '\nold predicate vector: ', predicate_vectors[idx],
                          '\nnew predicate vector: ', next_predicate_vectors[idx],
                          '\nutility value: ', states_util_values[idx],
                          '\nnext utility: ', next_state_util_values[idx])

        # clip the shaping reward for stabilization consideration
        shaping_rewards = self.hyper_params["SHAPING_REWARD_WEIGHT"] * np.clip(shaping_rewards,
                                                                            self.hyper_params["SHAPING_REWARD_CLIP"][0],
                                                                            self.hyper_params["SHAPING_REWARD_CLIP"][1])
        if self.hyper_params["NEGATIVE_REWARD_ONLY"]:
            shaping_rewards = np.clip(shaping_rewards,
                                      self.hyper_params["SHAPING_REWARD_CLIP"][0],
                                      0)
        return shaping_rewards

    def get_states_utility(self, states, predicate_vectors):
        """
        Compute single state utility based on predicate values and predicate weights
        :param states: (np.ndarray) the states (can be a single state or a nested list: [[state, state, ...], [state, state, ...], ...]
        :param predicate_vectors: np.array storing the predicate vectors
        """
        # if nested list of states (traj), sum up the utility values of states in the same trajectory
        energy = self.eval(states, predicate_vectors)
        utility_values = (-energy)
        return utility_values.detach().cpu().numpy()

    def get_utility_values_vectors(self, states, predicate_values):
        n_states = len(predicate_values)
        util_vector = np.zeros(shape=(n_states, len(self.predicate_keys)))
        if self.is_doing_pretrain:
            return util_vector

        states = self._preprocess_state(states)
        states = img2simpleStates(states, end=self.hyper_params["SIMPLE_STATES_SIZE"]-self.hyper_params["NUM_PREDICATES"])
        predicate_vectors = self.to_predicate_vectors(predicate_values)

        neg_predicate_weights = self.explainer(states)
        if torch.isnan(neg_predicate_weights).any():
            print('[ERROR] Predicate weight contains nan: ', neg_predicate_weights)
        predicate_weights = (-neg_predicate_weights).detach().cpu().numpy()
        predicate_weights = np.clip(predicate_weights, -100, 100)

        for i in range(n_states):
            for j, _ in enumerate(self.predicate_keys):
                idx = j
                if self.hyper_params["ONE_HOT_PREDICATES"]:
                    idx = 2 * j
                    util_vector[i][j] = (predicate_vectors[i][idx] * predicate_weights[i][idx] +
                                         predicate_vectors[i][idx + 1] * predicate_weights[i][idx + 1])
                else:
                    util_vector[i][j] = predicate_vectors[i][idx] * predicate_weights[i][idx]

        return util_vector

    def get_utility_values_dicts(self, states, predicate_values):
        """
        Compute the utility value of each predicate and return a dict storing utility values
        """
        util_vectors = self.get_utility_values_vectors(states, predicate_values)

        util_value_dict = []
        for i in range(len(util_vectors)):
            util_dict = {}
            for idx, key in enumerate(self.predicate_keys):
                util_dict[key] = util_vectors[i][idx]
            util_value_dict.append(util_dict)
        return util_value_dict

    def eval_explainer(self, states, predicate_vectors, next_states, next_predicate_vectors):
        negative_utility_values, rewards = self.eval(states, predicate_vectors, eval_r=True)
        next_negative_utility_values = self.eval(next_states, next_predicate_vectors)
        return -rewards - next_negative_utility_values + negative_utility_values

    def eval(self, states, predicate_vectors, eval_r=False):
        """
        Compute single state utility value based on predicate values and predicate weights
        :param states: (np.ndarray) the states (can be a single state or a nested list: [[state, state, ...], [state, state, ...], ...]
        :param predicate_vectors: torch.Tensor storing the predicate values
        """
        states = self._preprocess_state(states)
        states = img2simpleStates(states, end=self.hyper_params["SIMPLE_STATES_SIZE"]-self.hyper_params["NUM_PREDICATES"])
        neg_predicate_weights = self.explainer(states)

        if torch.isnan(neg_predicate_weights).any():
            print('[ERROR] Predicate weight contains nan: ', neg_predicate_weights)
        predicate_vectors = torch.from_numpy(predicate_vectors).type(torch.FloatTensor).to(device)
        negative_utility_values = torch.sum(neg_predicate_weights * predicate_vectors, dim=-1, keepdim=True)
        if eval_r:
            rewards = self.reward_net(states)
            return negative_utility_values, rewards
        else:
            return negative_utility_values

    def augment_states(self, states, predicate_values):
        """ augment the original observation with utility map """
        if self.is_doing_pretrain:
            return states
        else:
            return states

    def _add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(*transition)

    def augment_experience(self, experiences, predicates, next_predicates, is_to_tensor=True):
        states, actions, rewards, next_states, _ = experiences[:5]
        assert len(states) == len(predicates), 'the number of states and predicates should be the same'

        predicate_vectors = self.to_predicate_vectors(predicates)
        # predicate_util_vectors = self.get_utility_values_vectors(states, predicates)
        # # states_predicates_util = self._preprocess_predicates_states(augmented_states,
        # #                                                             np.concatenate((predicate_vectors,
        # #                                                                             predicate_util_vectors), axis=1))


        next_predicate_vectors = self.to_predicate_vectors(next_predicates)
        # next_predicate_util_vectors = self.get_utility_values_vectors(next_states, next_predicates)

        # if self.hyper_params["STATES_WITH_UTILITIES"]:
        #     states_predicates_util = simpleStates2img(states, predicate_util_vectors,
        #                                               start=self.hyper_params["SIMPLE_STATES_SIZE"],
        #                                               end=self.hyper_params["SIMPLE_STATES_SIZE"] + self.hyper_params[
        #                                                   "NUM_PREDICATES"])
        #     next_states_predicates_util = simpleStates2img(states, next_predicate_util_vectors, start=self.hyper_params["SIMPLE_STATES_SIZE"], end=self.hyper_params["SIMPLE_STATES_SIZE"]+self.hyper_params["NUM_PREDICATES"])
        #
        # else:
        #     states_predicates_util = states
        #     next_states_predicates_util = next_states
        #
        # shaping_rewards = self.get_shaping_reward(states, predicates, next_states, next_predicates)
        # augmented_rewards = np.array(shaping_rewards)

        with torch.no_grad():
            states, next_states = self.np2torchTensor([states, next_states])
            # compute energy (energy is negative utility)
            energy = self.eval_explainer(states, predicate_vectors, next_states,
                                         next_predicate_vectors)

            batch_log_p = torch.sum(-energy, dim=1, keepdim=True)
            batch_log_p = torch.clamp(batch_log_p, max=self.hyper_params["MAX_ENERGY"])
            if torch.isnan(batch_log_p).any():
                print('[ERROR] batch_log_p contains nan: ', batch_log_p)

            states_s = img2simpleStates(states, end=self.hyper_params["SIMPLE_STATES_SIZE"])

            if self.hyper_params["STATES_WITH_UTILITIES"]:
                batch_log_q = self.evaluate_state_action(torch.cat((states_s, predicate_util_vectors), dim=-1),
                                                         actions).detach()
            else:
                batch_log_q = self.evaluate_state_action(states_s, actions).detach()

            # batch_log_q = torch.from_numpy(log_q).type(torch.FloatTensor).to(device)
            if torch.isnan(batch_log_q).any():
                print('[ERROR] batch_log_q contains nan: ', batch_log_q)

            batch_log_p_q = (batch_log_q.exp() + batch_log_p.exp() + self.hyper_params["LOG_REG"]).log()
            if torch.isnan(batch_log_p_q).any():
                print('[ERROR] batch_log_p_q contain nan: ', batch_log_p_q)
                print('[ERROR] batch_log_q.exp(): ', batch_log_q.exp())
                print('[ERROR] batch_log_p.exp(): ', batch_log_p.exp())

            scores = torch.exp(batch_log_p - batch_log_p_q)
            # D_value = torch.log(scores) - torch.log(1-scores)
            # scores = scores.detach().cpu().numpy()
            final_rewards = scores * 5
        # if is_to_tensor:
        #     augmented_rewards = self._to_float_tensor(augmented_rewards)
        # else:
        #     states_predicates_util = states_predicates_util.detach().cpu().numpy()
        #     next_states_predicates_util = next_states_predicates_util.detach().cpu().numpy()

        return states, final_rewards, next_states


    def np2torchTensor(self, items):
        return [torch.from_numpy(item).type(
                torch.FloatTensor).to(device) for item in items]

    def update_model(self, experiences):
        """Train the model after each episode."""
        with torch.autograd.set_detect_anomaly(True):
            states, actions, rewards, next_states, dones, weights, indices, eps_d = (
                experiences
            )
            # # re-normalize the weights such that they sum up to the value of batch_size
            # weights = weights / torch.sum(weights) * float(states.shape[0])

            predicates, next_predicates = img2simpleStates(states, start=self.hyper_params["SIMPLE_STATES_SIZE"] -
                                                                         self.hyper_params["NUM_PREDICATES"],
                                                           end=self.hyper_params[
                                                               "SIMPLE_STATES_SIZE"]), img2simpleStates(next_states,
                                                                                                        start=
                                                                                                        self.hyper_params[
                                                                                                            "SIMPLE_STATES_SIZE"] -
                                                                                                        self.hyper_params[
                                                                                                            "NUM_PREDICATES"],
                                                                                                        end=
                                                                                                        self.hyper_params[
                                                                                                            "SIMPLE_STATES_SIZE"])
            states, rewards, next_states = self.augment_experience(experiences, predicates, next_predicates)

            if self.hyper_params["STATES_WITH_UTILITIES"]:
                simple_states, next_simple_states = img2simpleStates(states,
                                                       end=self.hyper_params["SIMPLE_STATES_SIZE"] + self.hyper_params[
                                                           "NUM_PREDICATES"]), img2simpleStates(next_states,
                                                                                                end=self.hyper_params[
                                                                                                        "SIMPLE_STATES_SIZE"] +
                                                                                                    self.hyper_params[
                                                                                                        "NUM_PREDICATES"])
            else:
                simple_states, next_simple_states = img2simpleStates(states,
                                                                     end=self.hyper_params["SIMPLE_STATES_SIZE"]), img2simpleStates(next_states,
                    end=self.hyper_params[
                            "SIMPLE_STATES_SIZE"])

            actions, dones, weights = self.np2torchTensor([actions, dones, weights])
            # simple_states, next_simple_states = states, next_states

            new_actions, log_prob, pre_tanh_value, mu, std = self.actor(simple_states)

            # train alpha
            if self.hyper_params["AUTO_ENTROPY_TUNING"]:
                alpha_loss = torch.mean(
                    (-self.log_alpha * (log_prob + self.target_entropy).detach()) * weights
                )

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.zeros(1)
                alpha = self.hyper_params["W_ENTROPY"]

            # Q function loss
            masks = 1 - dones
            gamma = self.hyper_params["GAMMA"]
            q_1_pred = self.qf_1(simple_states, actions)
            q_2_pred = self.qf_2(simple_states, actions)
            v_target = self.vf_target(next_simple_states)
            q_target = rewards + self.hyper_params["GAMMA"] * v_target * masks
            qf_1_loss = torch.mean((q_1_pred - q_target.detach()).pow(2) * weights)
            qf_2_loss = torch.mean((q_2_pred - q_target.detach()).pow(2) * weights)

            if self.use_n_step:
                experiences_n = self.memory_n.sample(indices)
                states, _, rewards, next_states, dones = experiences_n

                predicates, next_predicates = img2simpleStates(states,
                                                                   start=self.hyper_params["SIMPLE_STATES_SIZE"] -
                                                                         self.hyper_params["NUM_PREDICATES"],
                                                                   end=self.hyper_params[
                                                                       "SIMPLE_STATES_SIZE"]), img2simpleStates(
                    next_states,
                    start=
                    self.hyper_params[
                        "SIMPLE_STATES_SIZE"] -
                    self.hyper_params[
                        "NUM_PREDICATES"],
                    end=
                    self.hyper_params[
                        "SIMPLE_STATES_SIZE"])
                states, rewards, next_states = self.augment_experience(experiences_n, predicates,
                                                                             next_predicates)
                if self.hyper_params["STATES_WITH_UTILITIES"]:
                    simple_states, next_simple_states = img2simpleStates(states, end=self.hyper_params["SIMPLE_STATES_SIZE"] +
                                                                             self.hyper_params[
                                                                                 "NUM_PREDICATES"]), img2simpleStates(
                        next_states, end=self.hyper_params["SIMPLE_STATES_SIZE"] + self.hyper_params["NUM_PREDICATES"])

                else:
                    simple_states, next_simple_states = img2simpleStates(states,
                                                                         end=self.hyper_params["SIMPLE_STATES_SIZE"]), img2simpleStates(
                        next_states, end=self.hyper_params["SIMPLE_STATES_SIZE"])


                dones = self.np2torchTensor([dones])[0]
                # simple_states, next_simple_states = states, next_states
                gamma = gamma ** self.hyper_params["N_STEP"]
                lambda1 = self.hyper_params["LAMBDA1"]
                masks = 1 - dones

                v_target = self.vf_target(next_simple_states)
                q_target = rewards + gamma * v_target * masks
                qf_1_loss_n = torch.mean((q_1_pred - q_target.detach()).pow(2) * weights)
                qf_2_loss_n = torch.mean((q_2_pred - q_target.detach()).pow(2) * weights)

                # to update loss and priorities
                qf_1_loss = qf_1_loss + qf_1_loss_n * lambda1
                qf_2_loss = qf_2_loss + qf_2_loss_n * lambda1

            # V function loss
            v_pred = self.vf(simple_states)
            q_pred = torch.min(
                self.qf_1(simple_states, new_actions), self.qf_2(simple_states, new_actions)
            )
            v_target = (q_pred - alpha * log_prob).detach()
            vf_loss_element_wise = (v_pred - v_target).pow(2)
            vf_loss = torch.mean(vf_loss_element_wise * weights)

            if self.total_steps % self.hyper_params["DELAYED_UPDATE"] == 0:
                # actor loss
                advantage = q_pred - v_pred.detach()
                actor_loss_element_wise = alpha * log_prob - advantage
                actor_loss = torch.mean(actor_loss_element_wise * weights)

                # regularization
                mean_reg = self.hyper_params["W_MEAN_REG"] * mu.pow(2).mean()
                std_reg = self.hyper_params["W_STD_REG"] * std.pow(2).mean()
                pre_activation_reg = self.hyper_params["W_PRE_ACTIVATION_REG"] * (
                    pre_tanh_value.pow(2).sum(dim=-1).mean()
                )
                actor_reg = mean_reg + std_reg + pre_activation_reg

                # actor loss + regularization
                actor_loss += actor_reg

                # train actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # update target networks
                common_utils.soft_update(self.vf, self.vf_target, self.hyper_params["TAU"])

                # update priorities
                new_priorities = vf_loss_element_wise
                new_priorities += self.hyper_params[
                    "LAMBDA3"
                ] * actor_loss_element_wise.pow(2)
                new_priorities += self.hyper_params["PER_EPS"]
                new_priorities = new_priorities.data.cpu().numpy().squeeze()
                new_priorities += eps_d
                self.memory.update_priorities(indices, new_priorities)

                # increase beta
                fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)
            else:
                actor_loss = torch.zeros(1)


            # train Q functions
            self.qf_1_optimizer.zero_grad()
            qf_1_loss.backward()
            self.qf_1_optimizer.step()

            self.qf_2_optimizer.zero_grad()
            qf_2_loss.backward()
            self.qf_2_optimizer.step()

            # train V function
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

            return (
                actor_loss.data,
                qf_1_loss.data,
                qf_2_loss.data,
                vf_loss.data,
                alpha_loss.data,
            )

    def pretrain(self):
        """Pretraining steps."""
        pretrain_loss = list()
        print("[INFO] Pre-Train %d steps." % self.hyper_params["PRETRAIN_STEP"])
        for i_step in range(1, self.hyper_params["PRETRAIN_STEP"] + 1):
            experiences = self.memory.sample(beta=1.0)
            loss = self.update_model(experiences)
            pretrain_loss.append(loss)  # for logging

            # logging
            if i_step == 1 or i_step % 100 == 0:
                avg_loss = np.vstack(pretrain_loss).mean(axis=0)
                pretrain_loss.clear()
                self.write_log(
                    0, avg_loss, 0)#, delayed_update=self.hyper_params["DELAYED_UPDATE"]
                # )

    def step(self, action):
        """Take an action and return the response of the env."""
        self.total_steps += 1
        self.episode_steps += 1
        next_state, reward, done, _ = self.env.step(action)

        if not self.args.test:
            print("self.episode_steps == self.args.max_episode_steps", self.episode_steps, self.args.max_episode_steps)
            # if the last state is not a terminal state, store done as false
            # if self.sparse:
            #     done_bool = (
            #         False if reward != self.reached_goal_reward else True
            #     )
            # else:
            #     done_bool = (
            #         False if self.episode_steps == self.args.max_episode_steps else done
            #     )
            if make_data:
                self.experiment_recorder.add_transition(self.curr_state, action, reward, next_state, done)

            done_bool = (
                False if reward != self.reached_goal_reward else True
            )
            transition = (self.curr_state, action, reward, next_state, done_bool)
            self._add_transition_to_memory(transition)
            self.episode_transitions.append(transition)

        return next_state, reward, done

    def evaluate_state_action(self, states, actions):
        """
        Compute the probabilities/densities of the state-action pair.
        state: np.ndarray
        action: np.ndarray
        """
        _, log_probs, _, _, _ = self.actor(states)
        log_probs = torch.clamp(log_probs, min=np.log(self.hyper_params["LOG_REG"]), max=0.0)
        return log_probs

    # noinspection PyArgumentList
    def _update_explainer(self):
        """
        update the discriminator
        :param: indexed_sampled_traj: the sampled trajectories from current iteration
        """
        # calculate the good batch size
        n_good = len(self.good_buffer)
        good_batch_size = min(n_good, int(self.hyper_params["BATCH_SIZE"]))

        # sample batch_size
        n_bad = len(self.bad_buffer)
        bad_batch_size = min(n_bad, int(self.hyper_params["BATCH_SIZE"]))

        losses_iteration = []
        good_predict_acc = []
        bad_predict_acc = []
        for it in range(self.hyper_params["MULTIPLE_LEARN"]):
            # sample good trajectories
            good_indices = np.random.randint(low=0, high=n_good, size=good_batch_size)
            bad_indices = np.random.randint(low=0, high=n_bad, size=bad_batch_size)
            n_samples = min(good_indices.shape[0], bad_indices.shape[0])

            arrange = np.arange(n_samples)
            np.random.shuffle(arrange)
            mini_batch_size = int(self.hyper_params["MINI_BATCH_SIZE"])
            for i in range(arrange.shape[0] // mini_batch_size):
                start_idx = mini_batch_size * i
                end_idx = mini_batch_size * (i + 1)
                good_batch_index = good_indices[arrange[start_idx:end_idx]]
                bad_batch_index = bad_indices[arrange[start_idx:end_idx]]

                # get labels of each sample
                batch_good_labels = np.ones(shape=(good_batch_index.shape[0],), dtype=int)
                batch_bad_labels = np.zeros(shape=(bad_batch_index.shape[0],), dtype=int)
                batch_labels = torch.FloatTensor(
                    torch.from_numpy(np.concatenate((batch_good_labels, batch_bad_labels), axis=0)).float()).to(
                    device).unsqueeze(1)

                # get good samples
                good_states, good_actions, good_rewards, good_next_states, _ = self.good_buffer.sample(indices=good_batch_index, is_to_tensor=False)
                # good_predicates, _ = self.good_buffer.get_utility_info(indices=good_batch_index)
                good_predicates = img2simpleStates(good_states, start=self.hyper_params["SIMPLE_STATES_SIZE"] -
                                                                         self.hyper_params["NUM_PREDICATES"],
                                                           end=self.hyper_params[
                                                               "SIMPLE_STATES_SIZE"])
                good_next_predicates = img2simpleStates(good_next_states, start=self.hyper_params["SIMPLE_STATES_SIZE"] -
                                                                      self.hyper_params["NUM_PREDICATES"],
                                                   end=self.hyper_params[
                                                       "SIMPLE_STATES_SIZE"])

                # get bad samples
                bad_states, bad_actions, bad_rewards, bad_next_states, _ = self.bad_buffer.sample(indices=bad_batch_index, is_to_tensor=False)
                bad_predicates = img2simpleStates(bad_states, start=self.hyper_params["SIMPLE_STATES_SIZE"] -
                                                                         self.hyper_params["NUM_PREDICATES"],
                                                           end=self.hyper_params[
                                                               "SIMPLE_STATES_SIZE"])
                bad_next_predicates = img2simpleStates(bad_next_states, start=self.hyper_params["SIMPLE_STATES_SIZE"] -
                                                                    self.hyper_params["NUM_PREDICATES"],
                                                  end=self.hyper_params[
                                                      "SIMPLE_STATES_SIZE"])

                # get states
                batch_states = np.concatenate((good_states, bad_states), axis=0)
                batch_actions = np.concatenate((good_actions, bad_actions), axis=0)
                batch_predicates = np.concatenate((good_predicates, bad_predicates), axis=0)
                batch_predicate_vectors = self.to_predicate_vectors(batch_predicates)
                batch_rewards = np.concatenate((good_rewards, bad_rewards), axis=0)
                batch_next_states = np.concatenate((good_next_states, bad_next_states), axis=0)
                batch_next_predicates = np.concatenate((good_next_predicates, bad_next_predicates), axis=0)
                batch_next_predicate_vectors = self.to_predicate_vectors(batch_next_predicates)

                # augment feature
                info = None
                # if self.hyper_params.augmented_feature:
                #     predicate_util_vectors = self.get_utility_values_vectors(batch_states, batch_predicates)
                #     info = {TRAJECTORY_INDEX.PREDICATE_VECTOR.value: batch_predicate_vectors,
                #             TRAJECTORY_INDEX.UTILITY_VECTOR.value: predicate_util_vectors}



                img_batch_states = deepcopy(batch_states)
                batch_util_vectors = self.get_utility_values_vectors(batch_states, batch_predicates) # Compute utilities
                batch_states = img2simpleStates(batch_states, end=self.hyper_params["SIMPLE_STATES_SIZE"])
                batch_states, batch_util_vectors, batch_rewards, batch_actions = self.np2torchTensor([batch_states, batch_util_vectors, batch_rewards, batch_actions])

                # compute log_q
                # policy.evaluate_states_actions
                # add no_grad? .detach()
                if self.hyper_params["STATES_WITH_UTILITIES"]:
                    batch_log_q = self.evaluate_state_action(torch.cat((batch_states, batch_util_vectors), dim=-1), batch_actions).detach()
                else:
                    batch_log_q = self.evaluate_state_action(batch_states, batch_actions).detach()

                # batch_log_q = torch.from_numpy(log_q).type(torch.FloatTensor).to(device)
                if torch.isnan(batch_log_q).any():
                    print('[ERROR] batch_log_q contains nan: ', batch_log_q)

                # batch_states = batch_states[:, :-self.hyper_params["NUM_PREDICATES"]] # Remove the predicates part from the end

                # compute energy (energy is negative utility)
                energy = self.eval_explainer(img_batch_states, batch_predicate_vectors, batch_next_states, batch_next_predicate_vectors)

                batch_log_p = torch.sum(-energy, dim=1, keepdim=True)
                batch_log_p = torch.clamp(batch_log_p, max=self.hyper_params["MAX_ENERGY"])
                if torch.isnan(batch_log_p).any():
                    print('[ERROR] batch_log_p contains nan: ', batch_log_p)

                batch_log_p_q = (batch_log_q.exp() + batch_log_p.exp() + self.hyper_params["LOG_REG"]).log()
                if torch.isnan(batch_log_p_q).any():
                    print('[ERROR] batch_log_p_q contain nan: ', batch_log_p_q)
                    print('[ERROR] batch_log_q.exp(): ', batch_log_q.exp())
                    print('[ERROR] batch_log_p.exp(): ', batch_log_p.exp())

                # compute the loss
                loss = batch_labels * (batch_log_p - batch_log_p_q) + (1 - batch_labels) * (batch_log_q - batch_log_p_q)
                if torch.isnan(loss).any():
                    print('[ERROR] loss contain nan: ', loss)
                    print('[ERROR] batch_labels: ', batch_labels)
                    print('[ERROR] batch_log_p_q: ', batch_log_p_q)
                    print('[ERROR] batch_log_p: ', batch_log_p)
                    print('[ERROR] batch_log_q: ', batch_log_q)
                    print('[ERROR] batch_log_p - batch_log_p_q: ', batch_log_p - batch_log_p_q)
                    print('[ERROR] batch_log_q - batch_log_p_q: ', batch_log_q - batch_log_p_q)

                # maximize the log likelihood -> minimize mean loss
                mean_loss = -torch.mean(loss)
                self.explainer_optim.zero_grad()
                self.reward_net_optim.zero_grad()
                mean_loss.backward()

                if self.hyper_params["SE_GRAD_CLIP"] is not None:
                    clip_grad_norm_(self.explainer.parameters(), self.hyper_params["SE_GRAD_CLIP"])
                    clip_grad_norm_(self.reward_net.parameters(), self.hyper_params["SE_GRAD_CLIP"])

                self.explainer_optim.step()
                self.reward_net_optim.step()

                # for logging: get the discriminator output
                clone_log_p_tau = batch_log_p.clone().detach().cpu().numpy()
                clone_log_p_q = batch_log_p_q.clone().detach().cpu().numpy()
                n_batch_good = good_batch_index.shape[0]
                n_batch_bad = bad_batch_index.shape[0]
                good_prediction = np.exp(clone_log_p_tau[0:n_batch_good] - clone_log_p_q[0:n_batch_good])
                bad_prediction = np.exp(
                    clone_log_p_tau[n_batch_good:n_batch_good + n_batch_bad]
                    - clone_log_p_q[n_batch_good:n_batch_good + n_batch_bad])
                good_predict_acc.append(np.mean((good_prediction > 0.5).astype(float)))
                bad_predict_acc.append(np.mean((bad_prediction < 0.5).astype(float)))
                losses_iteration.append(mean_loss.item())

        return np.mean(losses_iteration), np.mean(good_predict_acc), np.mean(bad_predict_acc)

    def select_action(self, state):
        """Select an action from the input space."""
        # if initial random action should be conducted
        if (
            self.total_steps < self.hyper_params["INITIAL_RANDOM_ACTION"]
            and not self.args.test
        ):
            # unscaled_random_action = self.env.action_space.sample()
            # # Unscaled between true action_space bounds
            # return common_utils.reverse_action(unscaled_random_action, self.env.action_space)
            return self.env.action_space.sample()

        if self.args.test:
            _, _, _, selected_action, _ = self.actor(state)
        else:
            selected_action, _, _, _, _ = self.actor(state)

        return selected_action.detach().cpu().numpy()

    def train(self):
        """Train the agent."""

        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(self.hyper_params)
            wandb.config.update(vars(self.args))
            wandb.watch([self.actor, self.vf, self.qf_1, self.qf_2, self.explainer, self.reward_net], log="parameters")

        if self.hyper_params["IF_PRETRAIN_DEMO"]:
            self.is_doing_pretrain = True
            self.pretrain()
            self.is_doing_pretrain = False

        for i_episode in range(1, self.args.episode_num + 1):
            if self.total_steps >= self.args.max_total_steps:
                break
            t_begin = time.time()
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()
            self.episode_transitions = []
            self.episode_steps = 0

            # Update the self-explainer
            explainer_loss_info = self._update_explainer()

            if make_data:
                self.experiment_recorder = ExperimentLogger(self.args.demo_path, "push", save_trajectories=True)
                self.experiment_recorder.redirect_output_to_logfile_as_well()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                for i in range(self.hyper_params["MULTIPLE_LEARN"]):
                    if len(self.memory) >= self.hyper_params["BATCH_SIZE"]:
                        experiences = self.memory.sample(beta=self.beta) if "PER_BETA" in self.hyper_params else self.memory.sample()
                        # experiences = self.memory.sample()
                        loss = self.update_model(experiences)
                        loss_episode.append(loss)  # for logging

                predicates = img2simpleStates(state, start=self.hyper_params["SIMPLE_STATES_SIZE"] - self.hyper_params["NUM_PREDICATES"], end=self.hyper_params["SIMPLE_STATES_SIZE"])
                predicate_util_vectors = self.get_utility_values_vectors(state, [predicates])[0]
                self.curr_state = deepcopy(state)
                state = img2simpleStates(state, end=self.hyper_params["SIMPLE_STATES_SIZE"])
                if self.hyper_params["STATES_WITH_UTILITIES"]:
                    state = np.concatenate((state, predicate_util_vectors))
                state = torch.FloatTensor(state).to(device)
                action = self.select_action(state)
                # The step function calls NormalizedAction gym wrapper and automatically scales the action to action_space range
                next_state, reward, done = self.step(action)
                t_state = draw_predicates_on_img(self.curr_state, self.hyper_params["SIMPLE_STATES_SIZE"], (640, 480), reward, done, utilv=predicate_util_vectors)
                cv2.imwrite(self.train_img_folder + '/color_img_' + str(i_episode) + '_' + str(self.episode_steps) + '.jpg', t_state)
                print("epi and step: ", i_episode, self.episode_steps, action, done, reward)
                # start1 = time.time()

                state = next_state
                score += reward

                self.bad_buffer.extend(self.episode_transitions)

            if make_data:
                self.experiment_recorder.save_trajectories(self.hyper_params["SIMPLE_STATES_SIZE"],
                                                      is_separated_file=True, is_save_utility=False,
                                                      show_predicates=True)

            predicates = img2simpleStates(state, start=self.hyper_params["SIMPLE_STATES_SIZE"] - self.hyper_params[
                "NUM_PREDICATES"], end=self.hyper_params["SIMPLE_STATES_SIZE"])
            final_predicate_util_vectors = self.get_utility_values_vectors(state, [predicates])[0]
            t_next_state = draw_predicates_on_img(state, self.hyper_params["SIMPLE_STATES_SIZE"], (640, 480), reward, done, utilv=final_predicate_util_vectors)
            cv2.imwrite(self.train_img_folder + '/color_img_' + str(i_episode) + '_' + str(self.episode_steps) + '.jpg',
                        t_next_state)

            # logging
            self.avg_scores_window.append(score)
            avg_score_window = float(np.mean(list(self.avg_scores_window)))

            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                self.write_log(i_episode, avg_loss, score, self.hyper_params["DELAYED_UPDATE"], avg_score_window)


            t_end = time.time()
            iteration_time_cost = t_end - t_begin

            if i_episode % self.args.save_period == 0:
                params = {
                    "actor": self.actor.state_dict(),
                    "qf_1": self.qf_1.state_dict(),
                    "qf_2": self.qf_2.state_dict(),
                    "vf": self.vf.state_dict(),
                    "vf_target": self.vf_target.state_dict(),
                    "actor_optim": self.actor_optimizer.state_dict(),
                    "qf_1_optim": self.qf_1_optimizer.state_dict(),
                    "qf_2_optim": self.qf_2_optimizer.state_dict(),
                    "vf_optim": self.vf_optimizer.state_dict(),
                    "explainer_state_dict": self.explainer.state_dict(),
                    "explainer_optim_state_dict": self.explainer_optim.state_dict(),
                    "reward_net_state_dict": self.reward_net.state_dict(),
                    "reward_net_optim_state_dict": self.reward_net_optim.state_dict(),
                }

                if self.hyper_params["AUTO_ENTROPY_TUNING"]:
                    params["alpha_optim"] = self.alpha_optimizer.state_dict()

                self.save_params(i_episode, params)

            # logging
            if self.args.log and explainer_loss_info:
                discriminator_avg_loss, good_avg_acc, bad_avg_acc = explainer_loss_info
                # discriminator_log_values = (
                #     i_episode,
                #     discriminator_avg_loss,
                #     good_avg_acc,
                #     bad_avg_acc,
                #     iteration_time_cost
                # )
                wandb.log({'explainer_loss': discriminator_avg_loss,
                'good accuracy': good_avg_acc,
                'bad accuracy': bad_avg_acc}, step=i_episode)

        # termination
        self.env.close()

    def load_params(self, path):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path, map_location=torch.device('cpu'))
        self.actor.load_state_dict(params["actor"])
        self.qf_1.load_state_dict(params["qf_1"])
        self.qf_2.load_state_dict(params["qf_2"])
        self.vf.load_state_dict(params["vf"])
        self.vf_target.load_state_dict(params["vf_target"])
        self.actor_optimizer.load_state_dict(params["actor_optim"])
        self.qf_1_optimizer.load_state_dict(params["qf_1_optim"])
        self.qf_2_optimizer.load_state_dict(params["qf_2_optim"])
        self.vf_optimizer.load_state_dict(params["vf_optim"])
        self.explainer.load_state_dict(params["explainer_state_dict"])
        self.explainer_optim.load_state_dict(params["explainer_optim_state_dict"])
        self.reward_net.load_state_dict(params["reward_net_state_dict"])
        self.reward_net_optim.load_state_dict(params["reward_net_optim_state_dict"])

        if self.hyper_params["AUTO_ENTROPY_TUNING"]:
            self.alpha_optimizer.load_state_dict(params["alpha_optim"])

        print("[INFO] loaded the model and optimizer from", path)

    def test(self):
        self.is_doing_pretrain = False
        self.episode_steps = 0

        """Test the agent."""
        for i_episode in range(self.args.episode_num):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                predicates = img2simpleStates(state, start=self.hyper_params["SIMPLE_STATES_SIZE"] - self.hyper_params[
                    "NUM_PREDICATES"], end=self.hyper_params["SIMPLE_STATES_SIZE"])
                predicate_util_vectors = self.get_utility_values_vectors(state, [predicates])[0]
                self.curr_state = deepcopy(state)
                state = img2simpleStates(state, end=self.hyper_params["SIMPLE_STATES_SIZE"])
                if self.hyper_params["STATES_WITH_UTILITIES"]:
                    state = np.concatenate((state, predicate_util_vectors))
                state = torch.FloatTensor(state).to(device)
                action = self.select_action(state)
                # The step function calls NormalizedAction gym wrapper and automatically scales the action to action_space range
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward
                step += 1

            print(
                "[INFO] episode %d\tstep: %d\ttotal score: %d"
                % (i_episode, step, score)
            )

        # termination
        self.env.close()
