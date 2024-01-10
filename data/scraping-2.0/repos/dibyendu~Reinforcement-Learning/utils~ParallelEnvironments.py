'''
Modified from OpenAI Baselines code to work with multi-agent environments
https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
'''

import os
import sys
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe

###### openAI gym single-agent and multi-agent modules ######
import gym
import utils.multiagent.scenarios as scenarios
from utils.multiagent.environment import MultiAgentEnv
#############################################################

def tile_images(img_nhwc, show_border=True, border_width=4, border_color=0):
    '''
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    '''
    img_nhwc = np.asarray(img_nhwc)
    N, n_frames, h, w, c = img_nhwc.shape
    if show_border:
        bordered_img = np.full((N, n_frames, h + 2 * border_width, w + 2 * border_width, c), border_color, dtype=np.uint8)
        for i, img in enumerate(img_nhwc):        
            bordered_img[i,:,border_width:-border_width,border_width:-border_width:,] = img
        img_nhwc = bordered_img
    N, _, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

class CloudpickleWrapper(object):
    '''
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    '''
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VecEnv(ABC):
    '''
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    '''
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset_individual(self, index):
        '''
        Reset a single environment referred by index
        '''
        pass
    
    @abstractmethod
    def reset(self):
        '''
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        '''
        pass

    @abstractmethod
    def step_async(self, actions):
        '''
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        '''
        pass

    @abstractmethod
    def step_wait(self):
        '''
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        '''
        pass

    @abstractmethod
    def close(self):
        '''
        Clean up the environments' resources.
        '''
        pass

    def step(self, actions):
        '''
        Step the environments synchronously.
        This is available for backwards compatibility.
        '''
        self.step_async(actions)
        return self.step_wait()

    @abstractmethod
    def render(self, mode='rgb_array'):
        '''
        Return RGB images from each environment
        '''
        pass

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

class VecEnvWrapper(VecEnv):
    '''
    An environment wrapper that applies to an entire batch
    of environments at once.
    '''
    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(self, num_envs=venv.num_envs, observation_space=observation_space or venv.observation_space, action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='rgb_array'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()




class ParallelEnvBase(VecEnv):
    @abstractmethod
    def worker(self, remote, parent_remote, env_fn_wrapper):
        '''
        Worker method for each environment.
        Override this method in the derived class.
        Should be different for single-agent and multi-agent systems.
        
        To disable console output for child processes, use

            sys.stdout = sys.stderr = open(os.devnull, 'w')
        '''
        pass

    def __init__(self, env_fns):
        '''
        envs: list of gym environments to run in subprocesses
        '''
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=self.worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
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
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_individual(self, index):
        self.remotes[index].send(('reset', None))
        return self.remotes[index].recv()

    def render(self, mode='rgb_array'):
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        bigimg = tile_images(imgs)
        if mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

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

class ParallelEnv(ParallelEnvBase):

    def __init__(self, env_name, n_envs=1, seed=1):
        '''
        Create multiple parallel threads of a single-agent environment
        
        Params
        ======
            env_name (str): name of the environment
            n_envs (int)  : number of copies of the environment
            seed (int)    : base seed to initialize the environments with random seeds
        '''
        def get_env_fn(index):
            def init_env():
                env = gym.make(env_name)
                env.seed(seed + index * 1000)
                np.random.seed(seed + index * 1000)
                return env
            return init_env
        ParallelEnvBase.__init__(self, [get_env_fn(i) for i in range(n_envs)])

    def worker(self, remote, parent_remote, env_fn_wrapper):
        sys.stdout = sys.stderr = open(os.devnull, 'w') # disable console output for child processes
        parent_remote.close()
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
            elif cmd == 'reset_task':
                ob = env.reset_task()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError

class MultiAgentParallelEnv(ParallelEnvBase):

    def __init__(self, env_name, n_envs=1, seed=1):
        '''
        Create multiple parallel threads of a multi-agent environment
        
        Params
        ======
            env_name (str): name of the environment
            n_envs (int)  : number of copies of the environment
            seed (int)    : base seed to initialize the environments with random seeds
        '''
        def get_env_fn(index):
            def init_env():
                scenario = scenarios.load(env_name).Scenario()
                env = MultiAgentEnv(scenario.make_world(), scenario.reset_world, scenario.reward, scenario.observation)
                env.seed(seed + index * 1000)
                np.random.seed(seed + index * 1000)
                return env
            return init_env
        ParallelEnvBase.__init__(self, [get_env_fn(i) for i in range(n_envs)])
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()

    def worker(self, remote, parent_remote, env_fn_wrapper):
        sys.stdout = sys.stderr = open(os.devnull, 'w') # disable console output for child processes
        parent_remote.close()
        env = env_fn_wrapper.x()
        while True:            
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, ob_full, reward, done, info = env.step(data)
                if all(done):
                    ob = env.reset()
                remote.send((ob, ob_full, reward, done, info))
            elif cmd == 'reset':
                ob, ob_full = env.reset()
                remote.send((ob, ob_full))
            elif cmd == 'reset_task':
                ob = env.reset_task()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'get_agent_types':
                if all([hasattr(a, 'adversary') for a in env.agents]):
                    remote.send(['adversary' if a.adversary else 'agent' for a in env.agents])
                else:
                    remote.send(['agent' for _ in env.agents])
            elif cmd == 'get_agent_count':
                remote.send(env.n)
            else:
                raise NotImplementedError

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, obs_full, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(obs_full), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, obs_full = zip(*results)
        return np.stack(obs), np.stack(obs_full)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])
    
    @property
    def states(self):
        self.remotes[0].send(('get_spaces', None))
        observation_space, _ = self.remotes[0].recv()
        return observation_space
    
    @property
    def actions(self):
        self.remotes[0].send(('get_spaces', None))
        _, action_space = self.remotes[0].recv()
        return action_space
    
    @property
    def n_agents(self):
        self.remotes[0].send(('get_agent_count', None))
        return self.remotes[0].recv()
