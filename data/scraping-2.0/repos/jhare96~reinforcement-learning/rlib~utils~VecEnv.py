import numpy as np
import gym
import multiprocessing as mp
from itertools import chain

# Code was inspired from or modified from OpenAI baselines https://github.com/openai/baselines/tree/master/baselines/common

class Env(object):
    def __init__(self, env, worker_id=0): #, Wrappers=None, **wrapper_args):
        #self.env_id = env_id
        #env = gym.make(env_id)
        self.parent, self.child = mp.Pipe()
        self.worker = Worker(worker_id, env, self.child)
        self.worker.daemon = True
        self.worker.start()
        self.open = True        
    
    def __del__(self):
        self.close()
        self.parent.close()
        self.child.close()
    
    def __getattr__(self, name):
        attribute = self._send_step('getattr', name)
        return attribute()
        
    def _send_step(self,cmd,action):
        self.parent.send((cmd,action))
        return self._recieve
    
    def _recieve(self,):
        return self.parent.recv()

    def step(self,action, blocking=True):
        #if self.open:
        results = self._send_step('step', action)
        # if blocking:
        #     return results()
        # else:
        return results 
    
    def reset(self):
        #if self.open:
        results = self._send_step('reset', None)
        return results()
    
    def close(self):
        if self.open:
            self.open = False
            results = self._send_step('close', None)
            self.worker.join()
    
    def render(self):
        #if self.open:
        self._send_step('render', None)
    
class Worker(mp.Process):
    def __init__(self, worker_id, env, connection):
        import gym
        np.random.seed()
        mp.Process.__init__(self)
        self.env = env #gym.make(env_id)
        self.worker_id = worker_id
        self.connection = connection
    
    def _step(self):
        try:
            while True:
                cmd, a = self.connection.recv()
                if cmd == 'step':
                    obs, r, done, info = self.env.step(a)
                    # auto_reset moved to env wrappers 
                    self.connection.send((obs,r,done,info))
                elif cmd == 'render':
                    self.env.render()
                    #self.connection.send((1))
                elif cmd == 'reset':
                    obs = self.env.reset()
                    self.connection.send(obs)
                elif cmd == 'getattr':
                    self.connection.send(getattr(self.env, a))
                elif cmd == 'close':
                    self.env.close()
                    #self.connection.send((1))
                    break
        except KeyboardInterrupt:
            print("closing worker", self.worker_id)
        finally:
            self.env.close()
            #self.connection.close()

    def run(self,):
        self._step()



class BatchEnv(object):
    def __init__(self, env_constructor, env_id, num_envs, blocking=False, make_args={}, **env_args):
        #self.envs = [Env(env_constructor(gym.make(env_id),**env_args),worker_id=i) for i in range(num_envs)]
        self.envs = []
        for i in range(num_envs):
            env = gym.make(env_id, **make_args)
            self.envs.append(Env(env_constructor(env, **env_args)))
        #self.envs = [env_constructor(env_id=env_id,**env_args, worker_id=i) for i in range(num_envs)]
        self.blocking = blocking

    def __len__(self):
        return len(self.envs)
    
    def __getattr__(self, name):
        return getattr(self.envs[0], name)

    def step(self,actions):
        if self.blocking: # wait for each process to return results before starting the next
            results = [env.step(action,True) for env, action in zip(self.envs,actions)]
        else:
            results = [env.step(action,False) for env, action in zip(self.envs,actions)] # apply steps async
            results = [result() for result in results] # collect results
            
        obs, rewards, done, info = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(done), info
    
    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.stack(obs)
    
    def close(self):
        for env in self.envs:
            env.close()



def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

class ChunkEnv(object):
    def __init__(self, env_id, num_workers, num_chunks):
        self.num_workers = num_workers
        self.num_chunks = num_chunks
        self.env_id = env_id

        self.workers = []
        self.parents = []
        for i in range(num_workers):
            parent, child = mp.Pipe()
            worker = ChunkWorker(env_id,num_chunks,child)
            self.parents.append(parent)
            self.workers.append(worker)

        try:
            for worker in self.workers:
                worker.start()

        except KeyboardInterrupt:
            self.close()
            exit()
            #for w in self.workers:
                #w.env.close()
                #w.terminate()
                #exit()

        
    def _send_step(self,cmd,actions):
        for parent, action_chunk in zip(self.parents,chunks(actions, self.num_chunks)):
            parent.send((cmd,action_chunk))
        return self._recieve
    
    def _recieve(self,):
        return [parent.recv() for parent in self.parents]

    def step(self,actions,blocking=True):
        results = self._send_step('step', actions)
        if blocking:
            results = list(chain.from_iterable(results()))
            obs, rewards, dones, infos = zip(*results)
            return np.stack(obs), np.stack(rewards), np.stack(dones), infos
        else:
            return results
    
    def reset(self):
        results = self._send_step('reset',np.zeros((self.num_chunks*self.num_workers)))
        results = list(chain.from_iterable(results()))
        return np.stack(results)
    
    def close(self):
        results = self._send_step('close',np.zeros((self.num_chunks*self.num_workers)))
        for worker in self.workers:
            worker.join()

class ChunkWorker(mp.Process):
    def __init__(self, env_id, num_chunks, connection, render=False):
        mp.Process.__init__(self)
        self.envs = [gym.make(env_id) for i in range(num_chunks)]
        self.connection = connection
        self.render = render
        
    def run(self):
        while True:
            cmd, actions = self.connection.recv()
            if cmd == 'step':
                results = []
                for a, env in zip(actions,self.envs):
                    obs, r, done, info = env.step(a)
                    # auto_reset moved to env wrappers
                    if self.render:
                        self.env.render()
                    results.append((obs,r,done,info))
                self.connection.send(results)
            elif cmd == 'reset':
                results = []
                for a, env in zip(actions,self.envs):
                    obs = env.reset()
                    results.append(obs)
                self.connection.send(results)
            elif cmd == 'close':
                for env in self.envs:
                    env.close()
                self.connection.send((1))
                break


class DummyBatchEnv(object):
    def __init__(self, env_constructor, env_id, num_envs, make_args={}, **env_args):
        self.envs = [env_constructor(gym.make(env_id, **make_args),**env_args) for i in range(num_envs)]

    def __len__(self):
        return len(self.envs)

    def __getattr__(self, name):
        return getattr(self.envs[0], name)

    def step(self,actions):
        results = [env.step(action) for env, action in zip(self.envs,actions)]
        obs, rewards, done, info = zip(*results)
        return np.stack(obs).copy(), np.stack(rewards).copy(), np.stack(done).copy(), info
    
    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.stack(obs).copy()
    
    def close(self):
        for env in self.envs:
            env.close()