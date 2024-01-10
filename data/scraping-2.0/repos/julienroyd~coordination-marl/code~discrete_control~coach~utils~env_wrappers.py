"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
import matplotlib.pyplot as plt


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd.startswith('set_seed'):
            seed, rank = data
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            # Get info about env type (agent typesto distinguish between good guys and bad guys)
            # If the environment does not make this distinction, we set them all to 'agent'
            if hasattr(env, 'agents') and all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in env.agents])
            else:
                remote.send(['agent' for _ in range(env.nagents)])
        elif cmd == 'get_agent_colors':
            if hasattr(env, 'agents') and all([hasattr(a, 'color') for a in env.agents]):
                remote.send([a.color for a in env.agents])
            else:
                cm = plt.cm.get_cmap('tab20')
                remote.send([np.array(cm(float(i) / float(env.nagents))[:3]) for i in range(env.nagents)])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, name, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.name = name
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)

        # Create pipes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        # Create subprocesses
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        # Get info about env type (spaces, agent types)
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        # Get info about agents
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_colors', None))
        self.agent_colors = self.remotes[0].recv()

        # Initializes base class
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, env_i=None):
        # Resets a single environment (they might not terminate synchronously)
        if env_i is not None:
            self.remotes[env_i].send(('reset', None))
            return self.remotes[env_i].recv()
        # Resets all environments
        else:
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

    def seed(self, seed):
        for rank, remote in enumerate(self.remotes):
            remote.send(('set_seed', [seed, rank]))


class DummyVecEnv(VecEnv):
    """
    Fake Vector Environment that has the same interface as SubprocVecEnv
    It is used to support config.n_rollout_threads = 1
    """
    def __init__(self, env_fns, name):
        self.name = name
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]

        # Initializes base class
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)

        # Get info about env type (agent typesto distinguish between good guys and bad guys)
        # If the environment does not make this distinction, we set them all to 'agent'
        if hasattr(env, 'agents') and all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in env.agents]
        else:
            self.agent_types = ['agent' for _ in range(env.nagents)]

        if hasattr(env, 'agents') and all([hasattr(a, 'color') for a in env.agents]):
            self.agent_colors = [a.color for a in env.agents]
        else:
            cm = plt.cm.get_cmap('tab20')
            self.agent_colors = [np.array(cm(float(i) / float(env.nagents))[:3]) for i in range(env.nagents)]

        self.ts = np.zeros(len(self.envs), dtype='int')
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self, env_i=None):
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return

    def seed(self, seed):
        assert len(self.envs) == 1
        self.envs[0].seed(seed)
