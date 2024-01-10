# This code is from openai baseline, with slight modifications
# https://github.com/openai/baselines/tree/master/baselines/common/vec_env

from ctypes import Structure, c_double, c_int
from multiprocessing import Process, Pipe, Manager
from multiprocessing.sharedctypes import Value
import time

import numpy as np
import torch

from olympus.utils import debug
from olympus.utils.signals import SignalHandler


class WorkerStat(Structure):
    _fields_ = [
        ('step', c_int),
        ('reset', c_int),
        ('close', c_int),
        ('done', c_int),
        ('get_spaces', c_int),
        ('step_time', c_double),
        ('reset_time', c_double),
        ('duplicates', c_int)       # Number of duplicate states generated
    ]                               # if high this is a huge RED FLAG


def worker(remote, parent_remote, env_factory, stat: WorkerStat, unique_set):
    parent_remote.close()

    cmd, data = remote.recv()
    assert cmd == 'init', 'first message should be about environment init'
    env = env_factory(*data)

    def check_for_duplicates(state):
        """Check if the state is a duplicate"""
        key = hash(state.tostring())
        if key not in unique_set:
            unique_set[key] = 1
        else:
            stat.duplicates += 1

    while True:
        cmd, data = remote.recv()

        if cmd == 'step':
            stat.step += 1
            s = time.time()
            ob, reward, done, info = env.step(data)
            stat.step_time += time.time() - s

            if done:
                stat.done += 1
                stat.reset += 1
                s = time.time()
                ob = env.reset()
                stat.reset_time += time.time() - s

            check_for_duplicates(ob)
            remote.send((ob, reward, int(done), info))

        elif cmd == 'reset':
            stat.reset += 1

            s = time.time()
            ob = env.reset()
            stat.reset_time += time.time() - s

            check_for_duplicates(ob)
            remote.send(ob)

        elif cmd == 'reset_task':
            stat.reset += 1

            s = time.time()
            ob = env.reset_task()
            stat.reset_time += time.time() - s

            check_for_duplicates(ob)
            remote.send(ob)

        elif cmd == 'close':
            stat.close += 1
            debug('closing env')
            break

        elif cmd == 'get_spaces':
            stat.get_spaces += 1
            remote.send((env.observation_space, env.action_space))

        else:
            raise NotImplementedError

    debug('cleaning up')
    env.close()
    remote.close()


class VectorStat(Structure):
    _fields_ = [
        ('step', c_int),
        ('step_time', c_double)
    ]


class _ParallelEnvironmentCleaner(SignalHandler):
    def __init__(self, penv):
        super(_ParallelEnvironmentCleaner, self).__init__()
        self.env = penv

    def sigterm(self, signum, frame):
        return self.env.close()

    def sigint(self, signum, frame):
        return self.env.close()

    def atexit(self):
        return self.env.close()


# Create a manager for everybody
# Because it opens socket and does not close them
_manager = None


# TODO: Manage Seeding HERE
class ParallelEnvironment:
    """A group of environment that are computed in parallel"""

    def __init__(self, num_workers, transforms, env_factory, *env_args):
        global _manager
        if _manager is None:
            _manager = Manager()

        self.manager: Manager = _manager
        self.unique_set = self.manager.dict()

        self.worker_stat = Value(WorkerStat, 0, 0, 0, 0, 0, 0)
        self.vector_stat = VectorStat()
        self.remotes, work_remotes = zip(*[Pipe() for _ in range(num_workers)])

        self.ps = []
        for (work_remote, remote) in zip(work_remotes, self.remotes):
            self.ps.append(
                Process(target=worker, args=(work_remote, remote, env_factory, self.worker_stat, self.unique_set))
            )

        for p in self.ps:
            p.start()

        for remote in work_remotes:
            remote.close()

        for remote in self.remotes:
            remote.send(('init', env_args))

        self.waiting = False
        self.closed = False

        self.transforms = transforms
        if self.transforms is None:
            self.transforms = lambda x: x

        self.observation_space, self.action_space = self._get_spaces()

        # do not hang in case of error
        self._cleaner = _ParallelEnvironmentCleaner(self)
        self.dtype = torch.float

    def _get_spaces(self):
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        observation_space = self._to_shape(observation_space)
        action_space = self._to_shape(action_space)

        input = torch.randn((1,) + observation_space)
        observation_space = self.transforms(input).shape[1:]

        return observation_space, action_space

    def _to_shape(self, space):
        if isinstance(space, (tuple, torch.Size)):
            return space

        import gym
        if isinstance(space, gym.spaces.Discrete):
            return space.n,

        return space.shape

    @property
    def state_space(self):
        return self.observation_space

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        obs, rewards, dones, infos = zip(*results)
        return self.transforms(self._convert(np.stack(obs))),\
               self._convert(np.stack(rewards)),\
               self._convert(np.stack(dones)), infos

    def step(self, actions):
        assert self.waiting is not True, 'should call step_wait after step_async!'
        self.vector_stat.step += 1

        s = time.time()
        self.step_async(actions)
        r = self.step_wait()
        self.vector_stat.step_time += time.time() - s
        return r

    def report(self):
        stat = self.worker_stat
        avg = self.vector_stat.step_time / self.vector_stat.step
        stats = {
            'worker_step': stat.step,
            'worker_done': stat.done,
            'worker_reset': stat.reset,
            'worker_step_time': stat.step_time,
            'worker_reset_time': stat.reset_time,
            'worker_duplicates': stat.duplicates,
            'step_time_avg': avg,
            'unique_states': len(self.unique_set),
            'item/sec': len(self.remotes) / avg
        }

        return {
            'rl_loader': stats
        }

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
            p.close()

        for remote in self.remotes:
            remote.close()

        self.closed = True

    def _convert(self, x):
        return torch.from_numpy(
            np.stack(x)
        ).to(dtype=self.dtype)

    def render(self, mode='human'):
        pass

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

        return self.transforms(self._convert(np.stack([remote.recv() for remote in self.remotes])))

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))

        return self.transforms(self._convert(np.stack([remote.recv() for remote in self.remotes])))

    def get_spaces(self):
        return self.observation_space, self.action_space

    def __len__(self):
        return len(self.remotes)
