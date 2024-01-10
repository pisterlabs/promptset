"""
Modified from OpenAI Baselines code to work with multi-agent envs
https://github.com/marlbenchmark/on-policy
"""
import numpy as np
import torch
import torch.multiprocessing as mp
# from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
import env.chooseenv as ech
from env.olympics_running import OlympicsRunning
from rl_trainer.envs.Olympics_Env import OlympicsEnv
import platform
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

class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs):
        self.num_envs = num_envs
        # self.observation_space = observation_space
        # self.share_observation_space = share_observation_space
        # self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()



def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            # TODO: ob,reward,done, should be (num_agent,...) shape
            ob, reward, done, info = env.step(data)
            # if 'bool' in done.__class__.__name__:
            #     if done:
            #         ob = env.reset()
            # else:
            #     if np.all(done):
            #         ob = env.reset()

            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send((ob))
        # elif cmd == 'reset_task':
        #     ob = env.reset_task()
        #     remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        # elif cmd == 'get_spaces':
        #     remote.send((env.joint_action_space))
        elif cmd == "set_max_step":
            env.set_max_step(data)
            remote.send((None))
        else:
            raise NotImplementedError


class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        mp_type = "fork" if platform.system() == "Linux" else "spawn"
        ctx = mp.get_context(mp_type)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        # self.remotes[0].send(('get_spaces', None))
        # observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns))

    def step_async(self, actions):
        """
        actions: List[np.array([agent1_action,agent2_action])]
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos)

    def reset(self):
        """
        Returns:
            [num_rollout,...]
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)


    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def set_max_step(self,max_step,rollout_id):
        self.remotes[rollout_id].send(("set_max_step",max_step))
        self.remotes[rollout_id].recv()

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

def wrap_pytorch_task(env_setting: str,num_rollouts:int,rollout_maps=None,seed=0,shuffle_map=False,data_norm=False,use_astar=False):
    if rollout_maps is None:
        rollout_maps = [i%11+1 for i in range(num_rollouts)]
    assert len(rollout_maps) == num_rollouts,"[ERROR] map num is not equal to rollout number"


    def get_env_fn(seed,map_index):
        assert map_index in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        def init_env():
            environ: OlympicsRunning = ech.make(env_setting)
            environ.set_seed(seed)
            environ.specify_a_map(map_index)
            env = OlympicsEnv(environ,map_index,shuffle_map,data_norm,use_astar)
            return env
        return init_env

    return SubprocVecEnv([get_env_fn(seed+i*1000,rollout_maps[i]) for i in range(num_rollouts)])

if __name__ == '__main__':
    env = wrap_pytorch_task("olympics-running",
                            num_rollouts=4,
                            seed=0,
                            shuffle_map=True)
    env.reset()
    actions = [np.array([[0.,0.],[0.,0.]]) for _ in range(4)]
    for i in range(1):
        obs,rewards,dones,infos = env.step(actions)
        print(obs.shape)
        print(rewards.shape)
        print(dones.shape)
    env.close()