import contextlib
import os
from abc import ABC, abstractmethod
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn

"""
Copy-pasted from OpenAI to obviate dependency on Baselines. Required for vectorized environments.
"""

class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """

    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)


class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """

    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)


class VecEnv(ABC):
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
        bigimg = "ARGHH" #tile_images(imgs)
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

    
class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        super().__init__(num_envs=venv.num_envs,
                        observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space)

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

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.venv, name)

    
class VecEnvObservationWrapper(VecEnvWrapper):
    @abstractmethod
    def process(self, obs):
        pass

    def reset(self):
        obs = self.venv.reset()
        return self.process(obs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self.process(obs), rews, dones, infos

    
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

        
@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing
    Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)

        
class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs
    
class VecExtractDictObs(VecEnvObservationWrapper):
    def __init__(self, venv, key):
        self.key = key
        super().__init__(venv=venv,
            observation_space=venv.observation_space.spaces[self.key])

    def process(self, obs):
        return obs[self.key]
    
    
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

        
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)

        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        for i in range(len(infos)):
            infos[i]['env_reward'] = rews[i]
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)


class TransposeFrame(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(venv=env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.float32)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs.transpose(0,3,1,2), reward, done, info

    def reset(self):
        obs = self.venv.reset()
        return obs.transpose(0,3,1,2)


class ScaledFloatFrame(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(venv=env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs/255.0, reward, done, info

    def reset(self):
        obs = self.venv.reset()
        return obs/255.0

class MaskFloatFrame(VecEnvWrapper):
    def __init__(self, env, l):
        super().__init__(venv=env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
        self.l = l
        self.r = 0
        self.g = 0
        self.b = 0

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if self.l > 0:
            obs[:,0,:self.l,:] = self.r
            obs[:, 0, :, :self.l] = self.r
            obs[:, 0, -self.l:, :] = self.r
            obs[:, 0, :, -self.l:] = self.r

            obs[:, 1, :self.l, :] = self.g
            obs[:, 1, :, :self.l] = self.g
            obs[:, 1, -self.l:, :] = self.g
            obs[:, 1, :, -self.l:] = self.g

            obs[:, 2, :self.l, :] = self.b
            obs[:, 2, :, :self.l] = self.b
            obs[:, 2, -self.l:, :] = self.b
            obs[:, 2, :, -self.l:] = self.b
        return obs, reward, done, info

    def reset(self):
        obs = self.venv.reset()
        if self.l > 0:
            self.r = obs[0,0,63,0]
            self.g = obs[0,1,63,0]
            self.b = obs[0,2,63,0]

            obs[:,0,:self.l,:] = self.r
            obs[:, 0, :, :self.l] = self.r
            obs[:, 0, -self.l:, :] = self.r
            obs[:, 0, :, -self.l:] = self.r

            obs[:, 1, :self.l, :] = self.g
            obs[:, 1, :, :self.l] = self.g
            obs[:, 1, -self.l:, :] = self.g
            obs[:, 1, :, -self.l:] = self.g

            obs[:, 2, :self.l, :] = self.b
            obs[:, 2, :, :self.l] = self.b
            obs[:, 2, -self.l:, :] = self.b
            obs[:, 2, :, -self.l:] = self.b
        return obs

class MaskAllFrame(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(venv=env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

        brown = np.stack([191 * np.ones((64, 64)), 127 *  np.ones((64, 64)), 63 *  np.ones((64, 64))], axis=0)
        self.brown = np.expand_dims(brown, axis=0)


    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = obs.cpu()
        indexes = np.stack([(obs[:, 1, :, :] == 255), (obs[:, 1, :, :] == 255), (obs[:, 1, :, :] == 255)], axis=1)
        obs = obs * indexes + self.brown * (1 - indexes)

        return obs, reward, done, info

    def reset(self):
        obs = self.venv.reset().cpu()

        brown = np.repeat(self.brown, obs.shape[0], axis=0)

        indexes = np.stack([(obs[:, 1, :, :] == 255), (obs[:, 1, :, :] == 255), (obs[:, 1, :, :] == 255)], axis=1)
        obs = obs * indexes + brown * (1 - indexes)
        return obs


class MaskFrame(VecEnvWrapper):
    def __init__(self, env, l, device):
        super().__init__(venv=env)
        self.l = l
        self.device = device
        obs_shape = self.observation_space.shape

        brown = np.stack([191 * np.ones((64, 64)), 127 *  np.ones((64, 64)), 63 *  np.ones((64, 64))], axis=0)
        # self.brown = np.expand_dims(brown, axis=0)
        self.brown = torch.from_numpy(np.expand_dims(brown, axis=0)).float().to(self.device)
        self.conv = nn.Conv2d(1,1, 2*l+1, stride=1, padding =l, bias=False)
        self.conv.weight = nn.Parameter(torch.ones(1, 1, 2*l+1,2*l+1), requires_grad=False)
        self.conv.to(self.device)


    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()

        ind = 1.0*(obs[:, 1, :, :] == 255)
        ind = self.conv(ind.unsqueeze(dim=1))

        ind = (ind > 0.1)
        indexes = torch.cat((ind, ind, ind), dim=1)
        obs = obs * indexes + self.brown * torch.logical_not(indexes)


        return obs, reward, done, info

    def reset(self):
        obs = self.venv.reset()

        ind = 1.0*(obs[:, 1, :, :] == 255)
        ind = self.conv(ind.unsqueeze(dim=1))

        ind = (ind > 0.1)
        indexes = torch.cat((ind, ind, ind), dim=1)
        obs = obs * indexes + self.brown * torch.logical_not(indexes)

        return obs

