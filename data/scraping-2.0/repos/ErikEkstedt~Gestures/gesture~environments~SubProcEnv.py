import numpy as np
from multiprocessing import Process, Pipe

# from OpenAI-baselines ´baselines/common/vec_env/subproc_vec_env.py´
#
#       https://github.com/openai/baselines
#
# With added functionality for gesture

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


def worker_social(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            s, s_target, o, o_target, reward, done, info = env.step(data)
            if done:
                s, s_target, o, o_target = env.reset()
            remote.send((s, s_target, o, o_target, reward, done, info))
        elif cmd == 'reset':
            s, s_target, o, o_target = env.reset()
            remote.send((s, s_target, o, o_target))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.state_space, env.observation_space))
        elif cmd == 'render':
            remote.send(( env.render(data) ))
        elif cmd == 'set_target':
            remote.send(( env.set_target(data) ))
        else:
            raise NotImplementedError


class SubprocVecEnv_Social(object):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker_social, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.state_space, self.observation_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        state, s_target,  obs, o_target, rews, dones, infos = zip(*results)
        return np.stack(state), np.stack(s_target), \
                                np.stack(obs), \
                                np.stack(o_target), \
                                np.stack(rews), \
                                np.stack(dones), \
                                infos

    def render(self, modes):
        for remote, mode in zip(self.remotes, modes):
            remote.send(('render', mode))
        results = [remote.recv() for remote in self.remotes]
        human, machine, target = zip(*results)
        return np.stack(human), np.stack(machine), np.stack(target)

    def set_target(self, targets):
        for remote, target in zip(self.remotes, targets):
            remote.send(('set_target', target))
        results = [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        s, s_target, o, o_target = zip(*results)
        return np.stack(s), np.stack(s_target), np.stack(o), np.stack(o_target)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)
