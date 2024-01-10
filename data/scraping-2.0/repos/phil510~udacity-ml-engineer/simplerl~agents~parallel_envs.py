import abc
import numpy as np
import traceback
from multiprocessing import Process, Pipe, current_process

from marketenv import Wrapper

class Error(Exception):
    pass

class DeadProcessError(Error):
    pass

class CloudpickleWrapper(object):
    '''
    From OpenAI baselines
    Uses cloudpickle to serialize contents, otherwise multiprocessing 
    tries to use pickle
    '''
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def worker(remote, wrapped_env_fn):
    try:
        pid = current_process().pid
        env = wrapped_env_fn.x()
        remote.send((0, pid, None))
    except Exception as e:
        pid = current_process().pid
        remote.send((1, pid, traceback.format_exc()))
        remote.close()
        return
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                remote.send((0, pid, (obs, reward, done, info)))
            elif cmd == 'reset':
                obs = env.reset()
                remote.send((0, pid, obs))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_specs':
                try:
                    # not in the standard gym environment
                    env_spec = env.env_spec 
                except AttributeError:
                    env_spec = None
                remote.send((0, pid, (env.observation_space, 
                                      env.action_space,
                                      env.reward_range,
                                      env_spec)))
            elif cmd == 'getattr':
                if hasattr(env, data):
                    attr = getattr(env, data)
                    if callable(attr):
                        value = attr()
                    else:
                        value = attr
                    remote.send((0, pid, value))
                else:
                    raise AttributeError('Environment has no'
                                         + 'attribute "{}"'.format(data))
            else:
                raise NotImplementedError
    except Exception as e:
        remote.send((1, pid, traceback.format_exc()))
        remote.close()
    finally:
        env.close()
        
class ParallelEnvironment(object):
    '''
    Based on OpenAI baselines, simplified for the market environment(s) 
    used here
    '''
    def __init__(self, env_fns):
        self._env_fns = env_fns
        self._n_envs = len(env_fns)
        
        self._closed = False
        self._waiting = False
        
        # env remotes are kept in the main process
        # process remotes are send to the subprocesses
        self._env_remotes, self._process_remotes = zip(*[Pipe() for _ in 
                                                         range(self._n_envs)])
    
        self._ps = [Process(target = worker, 
                    args = (process_remote, CloudpickleWrapper(env_fn)))
                    for (process_remote, env_fn) 
                    in zip(self._process_remotes, self._env_fns)]
        for process in self._ps:
            # if the main process crashes, we should not cause things to hang
            process.daemon = True  
            process.start()
        
        for remote in self._env_remotes:
            _ = self._parse_response(remote.recv())
            
        self._env_remotes[0].send(('get_specs', None))
        o, a, r, s = self._parse_response(self._env_remotes[0].recv())
        self.observation_space = o
        self.action_space = a
        self.reward_range = r
        self.env_spec = s
    
    @property
    def closed(self):
        return self._closed
        
    def reset(self):
        self._assert_is_ready()
        
        for remote in self._env_remotes:
            remote.send(('reset', None))
        self._waiting = True
        
        obs = np.stack([self._parse_response(remote.recv()) 
                        for remote in self._env_remotes])
        self._waiting = False
        
        return obs
        
    def step(self, actions):
        self._assert_is_ready()
        
        for action, remote in zip(actions, self._env_remotes):
            remote.send(('step', action))
        self._waiting = True
        
        results = [self._parse_response(remote.recv()) 
                   for remote in self._env_remotes]
        self._waiting = False
        
        obs, reward, terminal, info = zip(*results)
        obs = np.stack(obs)
        reward = np.asarray(reward)
        terminal = np.asarray(terminal)
        info = list(info)
        
        return obs, reward, terminal, info
        
    def close(self):
        if self._closed:
            return
        
        self._closed = True
        if self._waiting:
            for remote in self._env_remotes:
                remote.recv()
                
        for remote in self._env_remotes:
            remote.send(('close', None))
            remote.close()
        
        for process in self._ps:
            process.join()
        
    def get_env_attr(self, attr):
        self._assert_is_ready()
        
        for remote in self._env_remotes:
            remote.send(('getattr', attr))
        self._waiting = True
        
        results = [self._parse_response(remote.recv()) 
                   for remote in self._env_remotes]
        
        self._waiting = False
        
        return results
        
    def _assert_is_ready(self):
        assert (not self._closed), 'Environment is closed'
        assert (not self._waiting), 'Waiting for responses from subprocesses'
        
    def _parse_response(self, response):
        if response[0] == 0:
            return response[2]
        else:
            for remote in self._env_remotes:
                remote.send(('close', None))
                remote.close()
            
            for process in self._ps:
                process.join()
            
            self._closed = True
            raise DeadProcessError('Error in child process'
                                   + '{}'.format(response[1])
                                   + ':\n' + response[2])
                                   
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
              
class VectorEnvWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        action = np.asarray(action).squeeze(0)
        obs, reward, terminal, info = self.env.step(action)
        if terminal:
            obs = self.env.reset()
        return np.expand_dims(obs, axis = 0), [reward], [terminal], [info]
        
    def reset(self):
        obs = self.env.reset()
        return np.expand_dims(obs, axis = 0)