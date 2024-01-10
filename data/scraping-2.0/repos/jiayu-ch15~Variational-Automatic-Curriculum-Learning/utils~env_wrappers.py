"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
# from baselines.common.vec_env import VecEnv, CloudpickleWrapper
import pdb

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

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

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

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

def simplifyworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if all(done):
                    ob = env.reset()           
            remote.send((ob, reward, done, info))
        elif cmd[0] == 'init_box_locking':
            starts_one = cmd[1]
            ob = env.reset(starts_one)
            remote.send(ob)
        elif cmd[0] == 'init_hidenseek':
            starts_one = cmd[1]
            ob = env.reset(starts_one)
            remote.send(ob)
        elif cmd == 'reset':
            ob = env.reset()         
            remote.send((ob))
        elif cmd == 'get_state':
            state = env.get_state()
            remote.send(state)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class SimplifySubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=simplifyworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def get_state(self): # the states of enities
        for remote in self.remotes:
            remote.send(('get_state', None))
        state = [remote.recv() for remote in self.remotes]
        return np.stack(state)

    def init_box_locking(self, starts, now_num_processes):
        i = 0
        results = []
        for remote in self.remotes:
            if i < now_num_processes:
                tmp_list = ['init_box_locking',starts[i]]
                remote.send((tmp_list, None))
                i += 1
        i = 0
        for remote in self.remotes:
            if i < now_num_processes:
                results.append(remote.recv())
                i += 1
        return np.stack(results)
    
    def init_hidenseek(self, starts, now_num_processes):
        i = 0
        results = []
        for remote in self.remotes:
            if i < now_num_processes:
                tmp_list = ['init_hidenseek',starts[i]]
                remote.send((tmp_list, None))
                i += 1
        i = 0
        for remote in self.remotes:
            if i < now_num_processes:
                results.append(remote.recv())
                i += 1
        return np.stack(results)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

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

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd[0] == 'step':
            # import time; start = time.time()
            actions = cmd[1]
            now_agent_num = cmd[2]
            ob, reward, done, info, available_actions = env.step(actions)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, available_actions = env.reset(now_agent_num)
            else:
                if all(done):
                    ob, available_actions = env.reset(now_agent_num) 
            remote.send((ob, reward, done, info, available_actions))
        elif cmd[0] == 'reset':
            now_agent_num = cmd[1]
            ob, available_actions= env.reset(now_agent_num)
            remote.send((ob, available_actions))
        elif cmd[0] == 'reset_pb':
            now_agent_num = cmd[1]
            now_box_num = cmd[2]
            ob, available_actions= env.reset(now_agent_num,now_box_num)
            remote.send((ob, available_actions))
        elif cmd[0] == 'set_initial_tasks_sp':
            now_agent_num = cmd[1]
            starts_one = cmd[2]
            ob = env.set_initial_tasks_sp(starts_one,now_agent_num)
            remote.send(ob)
        elif cmd[0] == 'set_initial_tasks_pb':
            now_agent_num = cmd[1]
            starts_one = cmd[2]
            ob = env.set_initial_tasks_pb(starts_one, now_agent_num)
            remote.send(ob)
        elif cmd[0] == 'new_starts_obs_sl':
            starts_one = cmd[1]
            ob = env.new_starts_obs_sl(starts_one)
            remote.send(ob)
        elif cmd == 'get_state':
            state = env.get_state()
            remote.send(state)
        elif cmd == 'get_goal':
            state, goal = env.get_state()
            remote.send((state,goal))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
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
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        self.length = len(env_fns)
        observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions, now_num_processes, now_agent_num):
        i = 0
        for remote, action in zip(self.remotes, actions):
            tmp_list = ['step', action, now_agent_num]
            if i < now_num_processes:
                remote.send((tmp_list,None))
                i += 1
        self.waiting = True

    def step_wait(self,now_num_processes):
        results = []
        i = 0
        for remote in self.remotes:
            if i < now_num_processes:
                results.append(remote.recv())
                i += 1
        self.waiting = False
        obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)
    
    def step(self, actions, now_num_processes, now_agent_num):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions, now_num_processes, now_agent_num)
        return self.step_wait(now_num_processes)

    def get_state(self): # the states of enities
        for remote in self.remotes:
            remote.send(('get_state', None))
        state = [remote.recv() for remote in self.remotes]
        return np.stack(state)
    
    def get_goal(self):
        for remote in self.remotes:
            remote.send(('get_goal', None))
        results = [remote.recv() for remote in self.remotes]
        state, goal = zip(*results)
        return np.stack(state), np.stack(goal)
        
    def reset(self, now_agent_num):
        # if now_box_num is None:
        #     for remote in self.remotes:
        #         remote.send((['reset',now_agent_num], None))
        # else:
        #     for remote in self.remotes:
        #         remote.send((['reset_pb',now_agent_num, now_box_num], None))
        for remote in self.remotes:
            remote.send((['reset',now_agent_num], None))
        results = [remote.recv() for remote in self.remotes]
        obs, available_actions = zip(*results)
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, self.length, observation_space, action_space)
        return np.stack(obs), np.stack(available_actions)

    def set_initial_tasks_sp(self, starts, now_agent_num, now_num_processes):
        i = 0
        results = []
        for remote in self.remotes:
            if i < now_num_processes:
                tmp_list = ['set_initial_tasks_sp', now_agent_num, starts[i]]
                remote.send((tmp_list, None))
                i += 1
        i = 0
        for remote in self.remotes:
            if i < now_num_processes:
                results.append(remote.recv())
                i += 1
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, self.length, observation_space, action_space)
        return np.stack(results)
        
    def set_initial_tasks_pb(self, starts, now_agent_num, now_num_processes):
        i = 0
        results = []
        for remote in self.remotes:
            if i < now_num_processes:
                tmp_list = ['set_initial_tasks_pb', now_agent_num, starts[i]]
                remote.send((tmp_list, None))
                i += 1
        i = 0
        for remote in self.remotes:
            if i < now_num_processes:
                results.append(remote.recv())
                i += 1
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, self.length, observation_space, action_space)
        return np.stack(results)
    
    def new_starts_obs_sl(self, starts, now_num_processes):
        i = 0
        results = []
        for remote in self.remotes:
            if i < now_num_processes:
                tmp_list = ['new_starts_obs_sl', starts[i]]
                remote.send((tmp_list, None))
                i += 1
        i = 0
        for remote in self.remotes:
            if i < now_num_processes:
                results.append(remote.recv())
                i += 1
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, self.length, observation_space, action_space)
        return np.stack(results)

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

def chooseworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info, available_actions = env.step(data)            
            remote.send((ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, available_actions = env.reset(data)           
            remote.send((ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class ChooseSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=chooseworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes,reset_choose):
            remote.send(('reset', choose))
        results = [remote.recv() for remote in self.remotes]
        obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(available_actions)

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
        
class DummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        ShareVecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos, available_actions = map(np.array, zip(*results))
        self.ts += 1
        
        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i], available_actions[i] = self.envs[i].reset()                   
                    self.ts[i] = 0
            else:
                if all(done):
                    obs[i], available_actions[i] = self.envs[i].reset()
                    self.ts[i] = 0
        
        self.actions = None

        return np.array(obs), np.array(rews), np.array(dones), infos, np.array(available_actions)

    def reset(self):  
        obs = []
        available_actions = []
        for env in self.envs:
            o,s = env.reset()
            obs.append(o)
            available_actions.append(s) 
        return np.array(obs), np.array(available_actions)

    def close(self):
        for env in self.envs:
            env.close()        