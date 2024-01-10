"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
import gym

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_attributes':
            attributes = env.get_attributes()
            remote.send(attributes)
        elif cmd == 'get_entities':
            entities = env.get_entities()
            remote.send(entities)
        elif cmd == 'get_observability':
            observability = env.get_observability()
            remote.send(observability)
        else:
            raise NotImplementedError

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        env = env_fns[0]()
        o = env.reset()
        n = len(o)
        dim_o = o[0].shape[-1]
        if gym.__version__ == "0.9.4":
            obs = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(n, dim_o))
        else:
            obs = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(n, dim_o), dtype=np.float32)
        act = gym.spaces.Discrete(5)
        self.nenvs = nenvs
        VecEnv.__init__(self, nenvs, obs, act)

    def get_attributes(self):
        for remote in self.remotes:
            remote.send(('get_attributes', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_entities(self):
        for remote in self.remotes:
            remote.send(('get_entities', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_observability(self):
        for remote in self.remotes:
            remote.send(('get_observability', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def step_async(self, actions):
        for i in range(self.nenvs):
            self.remotes[i].send(('step', actions[i]))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
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


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        o = env.reset()
        n = len(o)
        dim_o = o[0].shape[-1]
        if gym.__version__ == "0.9.4":
            obs = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(n, dim_o))
        else:
            obs = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(n, dim_o), dtype=np.float32)
        act = gym.spaces.Discrete(5)
        nenvs = 1
        VecEnv.__init__(self, nenvs, obs, act)

    def get_attributes(self):
        stats = [env.get_attributes() for env in self.envs]
        return np.array(stats)

    def get_entities(self):
        entities = [env.get_entities() for env in self.envs]
        return np.array(entities)

    def get_observability(self):
        entities = [env.get_observability() for env in self.envs]
        return np.array(entities)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        return np.array(obs), np.array(rews), np.array(dones), np.array(infos)

    def reset(self):
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return
