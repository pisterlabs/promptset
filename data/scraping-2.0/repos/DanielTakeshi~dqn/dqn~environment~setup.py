from dqn.environment.atari_wrappers import make_env
from dqn.environment import monitor
from dqn.replay.episode import Episode
from dqn.replay.transition import Transition
from dqn.utils.data_structures import SMAQueue
from dqn.utils.io import save_trajectory
import numpy as np
import logging, os, sys, time


class Environment:
    """Used as the `env` input to `replay.experience.ExperienceSource`."""

    def __init__(self, env_params, log_params, logger, seed, train=True):
        self.train = train
        self.seed = seed
        if train:
            self.flag = "train"
        else:
            self.flag = "test"
        self.logger = logger
        self.logger.debug("Initializing {0}ing environment".format(self.flag))

        # Wrapped 'env' that I usually think of from OpenAI, for stepping.
        self.env = make_env(
            env_name=env_params["name"],
            episode_life=(env_params["episode_life"] and train),
            clip_rewards=(env_params["clip_rewards"] and train),
            skip_frame=env_params["frame_skip"],
            frame_stack=env_params["frame_stack"],
            logdir=log_params["dir"])

        # Zero for games w/o lives (e.g. Pong). But can't guarantee episodes
        # use exaxctly this many.  https://github.com/CannyLab/dqn/issues/44
        self.lives_per_ep = (self.env.unwrapped).ale.lives()

        if train:
            logger.info("Environment: {0}".format(env_params["name"]))
            logger.info("Action space: {0}".format(env_params["num_actions"]))
            logger.info("Action meaning: {0}".format(env_params["action_meanings"]))
            logger.info("Observation space: {0}".format(env_params["obs_space"]))
            logger.info("Lives per episode: {}".format(self.lives_per_ep))
        self.env_params = env_params
        self.log_params = log_params
        self._life_idx = 0          # the LIFESPAN number, 1-indexed.
        self._episode_idx = 0       # the TRUE episode number, 1-indexed.
        self.total_steps = 0        # total steps taken by agent.
        self.summary = {}           # original code, for lifespans
        self.summary_true = {}      # for true episode statistics
        self._memory = None         # grayscale frames go here
        self._colored_memory = None # RGB color frames (don't use!)
        self._start_life_idx = 1    # extra stuff for `summary_true`
        self.start_time = None
        self.speed = None
        self.rewards_queue = SMAQueue(env_params["avg_window"])
        self.reset_episode()

    def reset(self):
        obs = self.env.reset()
        self.start_time = time.time()
        return obs

    def reset_episode(self):
        """Reset an episode and refresh memory.

        Because we use `self._memory` for saving frames of episodes. Now the
        next episode only contains frames from the upcoming, actual episode!

        Well, technically we reset to a new 'lifespan'. For Pong it's the same
        as a new episode. For Breakout, it means we lost a life here and will go
        to the next one (or a new episode if we actually lost all of them).
        Fortunately, in either case, we need `env.reset()`. Will add artificial
        steps to `monitor.csv`: https://github.com/CannyLab/dqn/issues/22.
        """
        self._life_idx += 1
        _obs = self.reset()
        self._memory = Episode(episode_num=self._life_idx, init_obs=_obs[0])
        if self.train and self.log_params["colored_output"]:
            self._colored_memory = Episode(episode_num=self._life_idx, init_obs=_obs[1])

    def finish_episode(self, save=False, gif=False, epsilon=None):
        """Book-keeping and debugging after finishing a life.

        Huge note: provides the statistics for the summaries that get used
        later for tracking episodes and lives. There are two summary
        dictionaries, one for (lives and clipped rewards) and the other for
        (true episodes and true reward).

        This is called after any `done` from `env.step` so for games with
        lives, this will get called after each life. It was originally designed
        for Pong, which is why the naming `episode` is still here. We report
        steps per second for throughput, and these are taken as scalar averages
        across each life. Actually, they are saved under 'frames_per_second'
        but it is really environment steps per second!
        """
        assert self.env_done
        self.summary[self._life_idx] = {
            "total_clip_rew": self.epi_rew,
            "steps": self.env_steps,
        }

        self.speed = self.env_steps / (time.time() - self.start_time)
        self.rewards_queue += self._memory.episode_total_reward
        _info = "Life {0}: {1}/{2} steps, {3:.2f} rewards, " \
                "{4:.2f} mean rewards, speed {5:.2f} frames/second".format(
                    self._life_idx, self.env_steps, self.total_steps,
                    self.epi_rew, self.mean_rew, self.speed)
        if epsilon:
            _info += ", epsilon {0:.2f}".format(epsilon)
        self.logger.debug(_info)
        if save:
            self.save_trajectory()
        if gif:
            self.save_trajectory_animation()

        # For 'true' episode stats, i.e., assuming 0 lives.
        true_results = monitor.load_results(self.log_params['dir'])
        true_rewards = true_results['r'].tolist()
        true_steps   = true_results['l'].tolist()
        if len(true_rewards) > self._episode_idx:
            # Let's 1-index the episode idx, like the life idx.
            self._episode_idx += 1
            self.summary_true[self._episode_idx] = {
                "raw_rew": true_rewards[-1],
                "steps": true_steps[-1],
                "life_idx_begin": self._start_life_idx,
                "life_idx_end": self._life_idx,
                "life_num": self._life_idx - self._start_life_idx + 1,
            }
            assert self._episode_idx == len(true_rewards)
            _info = "Episode {0} (1-idx) done, {1} steps, {2:.2f} raw reward".format(
                        self._episode_idx, true_steps[-1], true_rewards[-1])
            self._start_life_idx = self._life_idx + 1
            self.logger.debug(_info)

        # Again, reset the LIFESPAN, despite the naming here ...
        self.reset_episode()

    def step(self, _action):
        """Call usual gym step method.

        From running pong_standard_fast, in `dqn/replay/experience.py`, the
        stepping calls this method, except that `_memory.done` evaluates to be
        false, even when an episode finishes. (And it makes sense since if we
        called `finish_episode` here, the save argument is false, yet I keep
        seeing episodes saved...)

        The episodes are _actually_ saved (and `self.finish_episode(save=True)`
        called) back in the `dqn/replay/experience.py`'s __iter__ method since
        it checks if `self.env.env_done` and _that_ actually triggers saving.

        Careful about `_done`. Even if it's true, this might only indicate a
        loss of one life, and not the actual episode termination. But we are
        going to save like that anyway so it's OK.
        """
        _current_obs, _current_rew, _done, _info = self.env.step(_action)
        self.total_steps += 1
        transition = Transition(state=None, next_state=_current_obs[0],
                                action=_action, reward=_current_rew,
                                done=_done)
        self._memory.add_transition(transition)
        if self.train and self.log_params["colored_output"]:
            colored_transition = Transition(
                state=None, next_state=_current_obs[1],
                action=_action, reward=_current_rew, done=_done)
            self._colored_memory.add_transition(colored_transition)

    def save_trajectory(self):
        save_trajectory(
            dir_episodes=self.log_params["dir_episodes"],
            episode=self._life_idx,
            trajectory=self._memory,
            flag=self.flag)
        if self.train and self.log_params["colored_output"]:
            save_trajectory(
                dir_episodes=self.log_params["dir_episodes"],
                episode=self._life_idx,
                trajectory=self._colored_memory,
                flag="train_color")
        self.logger.debug("Life {0} saved: states "
            "length {1}, action length {2}, reward length {3}.".format(
                self._life_idx, self._memory.states.shape[0],
                self._memory.actions.shape[0],
                self._memory.rewards.shape[0]))

    def save_trajectory_animation(self):
        self._memory.to_animation(
            dir_episodes=self.log_params["dir_episodes"],
            flag=self.flag)
        self.logger.debug(
            "Life {0} trajectory GIF animation has been saved.".format(
                self._life_idx))

    def get_num_lives(self):
        return self._life_idx

    def get_num_episodes(self):
        return self._episode_idx

    @property
    def env_done(self):
        return self._memory.done

    @property
    def env_steps(self):
        return self._memory.length

    @property
    def env_obs(self):
        return self._memory.current_obs

    @property
    def env_rew(self):
        return self._memory.rew

    @property
    def mean_rew(self):
        return self.rewards_queue.mean()

    @property
    def epi_rew(self):
        assert self.env_done
        return self._memory.episode_total_reward

    @property
    def trajectory(self):
        assert self.env_done
        return self._memory
