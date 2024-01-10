"""This module is copied from openai baselines: https://github.com/openai/baselines"""
import numpy as np
from multiprocessing import Process, Pipe

class VecEnv(object):
    """
    Vectorized environment base class
    """
    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset all environments
        """
        raise NotImplementedError

    def close(self):
        pass


def worker(remote, env_fn_wrapper):
    if hasattr(env_fn_wrapper, 'x'):
        env = env_fn_wrapper.x()

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        elif cmd == 'render':
            env.render()
        elif cmd == 'init':
            env = env_fn_wrapper()
            # env.unwrapped.init(start_minecraft=True, videoResolution=(84, 84))
            # # _env.init(start_minecraft=True, videoResolution=(84, 84))
            # # wrap after minecraft has initiated
            # env = MinecraftWrapper(env)
            remote.send('finish_init')
        else:
            raise NotImplementedError


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
    def __init__(self, env_fns, minecraft=False):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        if minecraft:
            envs = env_fns
            self.ps = [Process(target=worker, args=(work_remote, env))
                       for (work_remote, env) in zip(self.work_remotes, envs)]
        else:
            self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
                for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]

        for p in self.ps:
            p.start()
        #
        if minecraft:
            self.init()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def init(self):
        for i, remote in enumerate(self.remotes):
            print('Sub process %s starts' % i)
            remote.send(('init', None))
            remote.recv()
            print('Sub process %s finishes' % i)

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def render(self):
        self.remotes[0].send(('render', None))

    @property
    def num_envs(self):
        return len(self.remotes)
