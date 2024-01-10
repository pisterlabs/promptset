#  Copyright: (C) ETAS GmbH 2019. All rights reserved.

#This code is from openai baseline
#https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import numpy as np
from multiprocess import Process, Pipe
from common.vec_env import VecEnvWrapper
# from baselines.common.running_mean_std import RunningMeanStd
from common.normalize import NormalizedActions


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if not isinstance(done, list):  # Check for Multi-Agent Environment
                if done:
                    ob = env.reset()
            remote.send((ob, reward, done, info))
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
        elif cmd == 'get_agents':
            remote.send(env.n_agents)
        else:
            raise NotImplementedError


class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    
class CloudpickleWrapper(object):
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

        
class SubprocVecEnv(VecEnv):
    """
    Vector environment for Multiprocessing of single-agent environments
    """
    def __init__(self, env_fns, spaces=None, use_norm=False, action_scale=7., action_scale_low=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        if use_norm:
            if action_scale_low is None:
                action_scale_low = -action_scale
            self.norm = NormalizedActions(low=action_scale_low, high=action_scale, use_numpy=True)
        else:
            self.norm = None
        self.do_reset = [False for i in range(self.nenvs)]
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        if self.norm is not None:
            actions = self.norm.reverse_normalize(actions)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

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
            
    def __len__(self):
        return self.nenvs


class BatchMultiVecEnv(VecEnv):
    """
    Vector environment for Multiprocessing of multi-agent environments
    Expects the API used by OpenAI (e.g. their particle environment
    see: https://github.com/openai/multiagent-particle-envs). In general a list with items corresponding to
    the environments. For actions as input and as output lists for observations, rewards and dones.
    """
    def __init__(self, env_fns, spaces=None, random_n_agents=True, use_norm=False, action_scale=7.):
        """
        envs: list of gym environments to run in subprocesses
        random_n_agents: If amount of agents is set randomly for each env. If not, iteraton time can be reduced.
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        self.n_agents = []
        self.get_n_agents()
        self.random_n_agents = random_n_agents
        if use_norm:
            self.norm = NormalizedActions(low=-action_scale, high=action_scale, use_numpy=True)
        else:
            self.norm = None

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        if self.norm is not None:
            actions = self.norm.reverse_normalize(actions)
        if self.random_n_agents:
            self.get_n_agents()
        actions = self.unpack_vector(actions)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def unpack_vector(self, action_vector):
        """
        action_vector: vector of actions or observations
        return: list of Vectors, each vector corresponds to one environment
        """
        action_list = []
        i = 0
        for n_agents in self.n_agents:
            action_list.append(action_vector[i:(i+n_agents)])
            i += n_agents
        return action_list

    def get_n_agents(self):
        self.n_agents = []
        for remote in self.remotes:
            remote.send(('get_agents', None))
            self.n_agents.append(remote.recv())
        return self.n_agents

    def step_wait(self):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        resets = []
        for remote in self.remotes:
            results = remote.recv()
            obs, reward, done, info = zip(results)
            obs_n.extend(obs[0])
            reward_n.extend(reward[0])
            done_n.extend(done[0])  #
            info_n.append(info[0])
            if "reset" in info[0]:
                resets.append(True)
            else:
                resets.append(False)
        self.waiting = False
        return np.stack(obs_n), np.stack(reward_n), np.stack(done_n), info_n

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs_n = []
        for remote in self.remotes:
            # obs_n += remote.recv()
            obs_n.extend(remote.recv())
        if self.random_n_agents:
            self.get_n_agents()
        return np.stack(obs_n)

    def reset_env(self, id):
        self.remotes[id].send(('reset', None))
        obs_n = []
        obs_n.extend(self.remotes[id].recv())
        return np.stack(obs_n)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        obs_n = []
        for remote in self.remotes:
            # obs_n += remote.recv()
            obs_n.extend(remote.recv())
        return np.stack(obs_n)

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

    def __len__(self):
        return self.nenvs*self.n_agents


class VecMultiNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, n_agents=2):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs*n_agents)
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_agents = n_agents

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs*self.n_agents)
        obs = self.venv.reset()
        return self._obfilt(obs)
