import torch
from torch.multiprocessing.queue import Queue

from custom_logging.collectibles import Collectibles
from custom_logging.utils.enums import Originator
from functools import partial
from marl.components.episode_batch import EpisodeBatch
from steppers.env_stepper import EnvStepper

from steppers.utils.env_worker_process import EnvWorker
from steppers.utils.stepper_utils import get_policy_team_id


class ParallelStepper(EnvStepper):
    def __init__(self, args, logger):
        """
        Based (very) heavily on SubprocVecEnv from OpenAI Baselines
        https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
        Runs multiple environments in parallel to play and collect episode batches to feed into a single learner.
        :param args:
        :param logger:
        """
        super().__init__(args, logger)
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        # Find id of the first policy team - Only supported for one policy team in the build plan
        teams = args.env_args["match_build_plan"]
        self.policy_team_id = get_policy_team_id(teams)

        # Make subprocesses for the envs
        self.in_queues, self.out_queues = zip(*[(Queue(), Queue()) for _ in range(self.batch_size)])
        self.workers = [
            EnvWorker(args, in_q=in_q, out_q=out_q)
            for (in_q, out_q) in zip(self.in_queues, self.out_queues)
        ]
        for worker in self.workers:
            worker.daemon = True
            worker.start()

        self.in_queues[0].put(("get_env_info", None))
        self.env_info = self.out_queues[0].get()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
        self.new_batch_fn = None
        self.scheme = None
        self.groups = None
        self.preprocess = None
        self.env_steps_this_run = 0

        self.home_mac = None
        self.home_batch = None

    def initialize(self, scheme, groups, preprocess, home_mac, away_mac=None):
        self.new_batch_fn = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                    preprocess=preprocess, device=self.args.device)
        self.home_mac = home_mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for in_q in self.in_queues:
            in_q.put(("close", None))

    def reset(self):
        self.home_batch = self.new_batch_fn()

        # Reset the envs
        for in_q in self.in_queues:
            in_q.put(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for out_q in self.out_queues:
            data = out_q.get()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.home_batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        """
        Run a single episode with multiple environments in parallel
        :param test_mode:
        :return:
        """
        self.reset()

        self.logger.test_mode = test_mode

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        eps = [0 for _ in range(self.batch_size)]
        self.home_mac.init_hidden(batch_size=self.batch_size)
        # bools to determine finished envs
        terminateds = [False for _ in range(self.batch_size)]
        # IDs of running envs
        running_envs = [idx for idx, terminated in enumerate(terminateds) if not terminated]
        env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        actions_batch = torch.zeros((self.batch_size, self.env_info["n_agents"]))

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions, is_greedy = self.home_mac.select_actions(self.home_batch, t_ep=self.t, t_env=self.t_env,
                                                              bs=running_envs,
                                                              test_mode=test_mode)

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }

            self.home_batch.update(actions_chosen, bs=running_envs, ts=self.t, mark_filled=False)

            # Send actions to each running env
            action_idx = 0
            for idx, in_q in enumerate(self.in_queues):
                if idx in running_envs:  # We produced actions for this env
                    if not terminateds[idx]:  # Only send the actions to the env if it hasn't terminated
                        actions_batch[action_idx] = actions[action_idx]
                        in_q.put(("step", actions_batch[action_idx]))
                    action_idx += 1  # actions is not a list over every env

            # Update running envs
            running_envs = [idx for idx, terminated in enumerate(terminateds) if not terminated]
            if all(terminateds):
                break  # all envs terminated -> end parallel episode

            post_transition_data = {  # Post step data we will insert for the current timestep
                "reward": [],
                "terminated": []
            }

            pre_transition_data = {  # Data for the next step we will insert in order to select an action
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            for idx, out_q in enumerate(self.out_queues):  # Receive step data back for each unterminated env
                if not terminateds[idx]:
                    data = out_q.get()
                    # Remaining data for this current timestep
                    policy_team_reward = data["reward"][0]  # ! Only supported if one policy team is playing
                    post_transition_data["reward"].append((policy_team_reward,))

                    episode_returns[idx] += policy_team_reward
                    eps[idx] += 1

                    if not test_mode:
                        self.env_steps_this_run += 1

                    done_n = data["terminated"]  # list of done booleans per team
                    terminated = any(done_n)
                    if terminated:  # if any team is done -> env terminated -> attach additional episode infos
                        env_infos.append(data["info"])
                    terminateds[idx] = terminated
                    post_transition_data["terminated"].append((terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transition data into the batch
            self.home_batch.update(post_transition_data, bs=running_envs, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.home_batch.update(pre_transition_data, bs=running_envs, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Send data collected during the episode - this data needs further processing
        self.logger.collect(Collectibles.RETURN, episode_returns, origin=Originator.HOME, parallel=True)
        self.logger.collect(Collectibles.WON, [env_info["battle_won"][0] for env_info in env_infos],
                            origin=Originator.HOME, parallel=True)
        self.logger.collect(Collectibles.WON, [env_info["battle_won"][1] for env_info in env_infos],
                            origin=Originator.AWAY, parallel=True)
        self.logger.collect(Collectibles.DRAW, [env_info["draw"] for env_info in env_infos], parallel=True)
        self.logger.collect(Collectibles.STEPS, self.t, parallel=True)
        # Log collectibles if conditions suffice
        self.logger.log(self.t_env)

        return self.home_batch, env_infos

    def __del__(self):
        # Close env workers
        self.close_env()
