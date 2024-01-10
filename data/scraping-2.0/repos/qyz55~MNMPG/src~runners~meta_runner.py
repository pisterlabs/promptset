from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import random
import torch as th

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class MetaRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
                            for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.new_batch_single = partial(EpisodeBatch, scheme, groups, 1, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess


    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0
    def reset_first(self):
        self.batch = self.new_batch_single()
        self.parent_conns[0].send(("reset", None))
        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        data = self.parent_conns[0].recv()
        pre_transition_data["state"].append(data["state"])
        pre_transition_data["avail_actions"].append(data["avail_actions"])
        pre_transition_data["obs"].append(data["obs"])
        self.batch.update(pre_transition_data, ts=0)
        self.t = 0
    def run(self, test_mode=False, meta_mode=False, use_rode=False):
        self.reset_first()
        episode_return = 0.0
        episode_length = 0
        if self.args.q_net_ensemble:
            chosen_index = random.randint(0,self.args.ensemble_num-1)
            chosen_mac = self.mac[chosen_index]
        else:
            chosen_mac = self.mac
        chosen_mac.init_hidden(batch_size=1)
        if self.args.mac == "separate_mac" or self.args.mac == "hierarchical_mac" or self.args.use_roma:
            chosen_mac.init_latent(batch_size=1)
        terminated = False
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        if meta_mode:
            log_ps=[]
        while True:
            ra = chosen_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, need_log_p=meta_mode)
            if meta_mode:
                if type(ra[-1]) == tuple:
                    if ra[-1][0] is not None:
                        log_ps.append(ra[-1][0]+ra[-1][1])
                    else:
                        log_ps.append(ra[-1][1])
                else:
                    log_ps.append(ra[-1])
            if use_rode:
                action, roles, role_avail_actions = ra[:3]
            elif meta_mode:
                action = ra[0]
            else:
                action= ra
            cpu_action = action.to("cpu").numpy()

            # Update the actions taken
            action_chosen = {
                "actions": action.unsqueeze(1),
            }
            if use_rode:
                action_chosen.update({
                    "roles": roles.unsqueeze(0).unsqueeze(1),
                    "role_avail_actions": role_avail_actions.unsqueeze(1)
                })
            self.batch.update(action_chosen, ts=self.t, mark_filled=False)

            # Send actions to each env
            if terminated:
                break
            self.parent_conns[0].send(("step", cpu_action[0]))
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            data = self.parent_conns[0].recv()
            post_transition_data["reward"].append((data["reward"],))
            episode_return += data["reward"]
            episode_length += 1
            env_terminated = False
            if data["terminated"] and not data["info"].get("episode_limit", False):
                env_terminated = True
            terminated = data["terminated"]
            post_transition_data["terminated"].append((env_terminated,))
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            self.batch.update(post_transition_data, ts=self.t, mark_filled=False)
            self.t += 1
            self.batch.update(pre_transition_data, ts=self.t, mark_filled=True)
        
        # collect log p for meta policy gradient
        if meta_mode:
            all_log_p = th.cat([it.unsqueeze(1) for it in log_ps[:-1]], dim=1) #[8*max_ep_len*3]
            batch_log_p = th.sum(all_log_p, [1,2])/all_log_p.size(1)
        if not test_mode:
            self.t_env += self.t

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = episode_length + cur_stats.get("ep_length", 0)

        cur_returns.append(episode_return)

        n_test_runs = max(1, self.args.test_nepisode)
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(chosen_mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", chosen_mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        final_reward = [episode_return/episode_length] if self.args.use_step_reward else episode_return
        if meta_mode:
            return self.batch, batch_log_p, final_reward
        else:
            return self.batch, final_reward
        
    def run_meta(self, test_mode=False, meta_mode=False, use_rode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        if self.args.q_net_ensemble:
            chosen_index = random.randint(0,self.args.ensemble_num-1)
            chosen_mac = self.mac[chosen_index]
        else:
            chosen_mac = self.mac
        chosen_mac.init_hidden(batch_size=self.batch_size)
        if self.args.mac == "separate_mac" or self.args.mac == "hierarchical_mac" or self.args.use_roma:
            chosen_mac.init_latent(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        if meta_mode:
            log_ps=[]
        while True:
            ra = chosen_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode, need_log_p=meta_mode)
            if meta_mode:
                if type(ra[-1]) == tuple:
                    if ra[-1][0] is not None:
                        log_ps.append(ra[-1][0]+ra[-1][1])
                    else:
                        log_ps.append(ra[-1][1])
                else:
                    log_ps.append(ra[-1])
            if use_rode:
                actions, roles, role_avail_actions = ra[:3]
                roles = roles.view(self.batch_size, -1)
            elif meta_mode:
                actions = ra[0]
            else:
                actions = ra
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            
            actions_chosen = {"actions": actions.unsqueeze(1)}
            if use_rode:
                actions_chosen.update({
                    "roles": roles[envs_not_terminated].unsqueeze(1),
                    "role_avail_actions": role_avail_actions[envs_not_terminated].unsqueeze(1)
                })
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)
        
        # collect log p for meta policy gradient
        if meta_mode:
            all_log_p = th.cat([it.unsqueeze(1) for it in log_ps[:-1]], dim=1) #[8*max_ep_len*3]
            ind = th.zeros([self.batch_size, max(episode_lengths)], device=self.batch.device)
            for i in range(self.batch_size):
                ind[i, :episode_lengths[i]] = 1.0
            batch_log_p = th.sum(all_log_p * ind.unsqueeze(2), [1,2])/th.sum(ind, 1)
            

        if not test_mode:
            self.t_env += self.t

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(chosen_mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", chosen_mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        final_reward = [i/j for i,j in zip(episode_returns, episode_lengths)] if self.args.use_step_reward else episode_returns
        if meta_mode:
            return self.batch, batch_log_p, final_reward
        else:
            return self.batch, final_reward

    def get_log_p(self, buffer):
        if self.args.q_net_ensemble:
            chosen_index = random.randint(0,self.args.ensemble_num-1)
            chosen_mac = self.mac[chosen_index]
        else:
            chosen_mac = self.mac
        chosen_mac.init_hidden(batch_size=buffer.batch_size)
        if self.args.use_roma:
            chosen_mac.init_latent(buffer.batch_size)
        buffer.to(self.batch.device)
        log_ps = []
        terminated = th.zeros(buffer.batch_size, device=self.batch.device)
        ind = th.zeros([buffer.batch_size, buffer.max_seq_length], device=self.batch.device)
        max_ep_len = 0
        for i in range(buffer.max_seq_length):
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if termed < 0.01]
            ra = chosen_mac.select_actions(buffer, t_ep=i, t_env=self.t_env, bs=envs_not_terminated, test_mode=False, need_log_p=True)
            log_p = ra[-1]
            if type(log_p) == tuple:
                if log_p[0] is not None:
                    log_p = log_p[0]+log_p[1]
                else:
                    log_p = log_p[1]
            log_ps.append(log_p)
            ind[~(terminated.round().to(th.bool)), i] = 1.0
            terminated += buffer["terminated"][:, i, 0]
            if sum(terminated).round().item()==buffer.batch_size:
                max_ep_len = i+1
                break
        if max_ep_len == 0:
            raise Exception("Some episodes have no 'terminated' mark.")
        ind = ind[:, :max_ep_len]
        all_log_p = th.cat([it.unsqueeze(1) for it in log_ps], dim=1) #[32*max_ep_len*3]
        # for i in range(self.batch_size):
        #     ind[i, :episode_lengths[i]] = 1.0
        batch_log_p = th.sum(all_log_p * ind.unsqueeze(2), [1,2])/th.sum(ind, 1)
        return batch_log_p

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

