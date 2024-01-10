# taken from openai/baselines: https://github.com/openai/baselines
# source: Deep Reinforcement Learning Nanodegree, Udacity.

import numpy as np
import gymnasium as gym
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod

class CloudpickleWrapper(object):
    """
    A wrapper that uses cloudpickle to serialize the contents, as multiprocessing
    typically uses pickle by default.
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VecEnv(ABC):
    """
    An abstract class representing an asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all environments and return an array of observations or a
        dictionary of observation arrays.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Instruct all environments to take a step with the given actions.
        Call step_wait() to get the results of the step.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the completion of the step taken with step_async().
        Returns (obs, rews, dones, infos).
        """
        pass

    @abstractmethod
    def close(self):
        """
        Release the resources used by the environments.
        """
        pass

    def step(self, actions):
        """
        Take a step in the environments synchronously.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        pass
        
    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, truncated, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, truncated, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class parallelEnv(VecEnv):
    def __init__(self, env_name='ALE/Pong-v5',
                 n=4, seed=None,
                 spaces=None):

        # Create gym environments
        env_fns = [gym.make(env_name) for _ in range(n)]

        # Set seeds for environments, if provided
        if seed is not None:
            for i, e in enumerate(env_fns):
                e.seed(i + seed)

        # Initialize multiprocessing components
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # Ensure subprocesses exit if the main process crashes
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, truncates, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(truncates), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True