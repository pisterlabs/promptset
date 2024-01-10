# Modified by Kenneth Marino
# VecNormalize originally copied from https://github.com/openai/baselines/
from vec_env import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
import numpy as np
import pdb

# From openai baselines originally
# My version of this saves the unclipped/unnormalized values for logging and other purposes
class ObservationFilter(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, train=True, noclip=False, has_timestep=False, ignore_mask=None, freeze_mask=None, time_scale=1e-3, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.train = train
        self.gamma = gamma
        self.epsilon = epsilon
        self.noclip = noclip
        self.ignore_mask = ignore_mask
        self.freeze_mask = freeze_mask
        self.has_timestep = has_timestep
        self.time_scale = time_scale

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        self.raw_obs = obs
        self.raw_rews = rews
        # Do filtering (but not for step_mask = 0 values)
        for proc, obs_proc in enumerate(obs):
            obs_proc = np.array([obs_proc])
            if self.step_mask[proc] > 0:
                obs_proc = self._obfilt(obs_proc)
            obs[proc] = obs_proc[0]
        if self.ret_rms:
            # Only update ret_rms if in training mode
            if self.train:
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            # Only update ob_rms if in training mode
            if self.train:
                # Use freeze mask to only update part of the ob_rms
                if self.freeze_mask is not None:
                    old_obs_rms_mean = np.array(self.ob_rms.mean)
                    old_obs_rms_var = np.array(self.ob_rms.var)   
                    self.ob_rms.update(obs)
                    self.ob_rms.mean = old_obs_rms_mean * self.freeze_mask + self.ob_rms.mean * (1 - self.freeze_mask)
                    self.ob_rms.var = old_obs_rms_var * self.freeze_mask + self.ob_rms.var * (1 - self.freeze_mask) 
                else:
                    self.ob_rms.update(obs)

            # Copy original obs
            obs_orig = np.copy(obs)

            # Use code from https://github.com/pat-coady/trpo/blob/5ac6b2e8476d0f1639a88128f59e8a51f1f8bce1/src/train.py#L92
            if self.noclip:
                obs = (obs - self.ob_rms.mean) / (3*(np.sqrt(self.ob_rms.var) + 0.1))
            else:
                obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob) 

            # Use ignore_mask to restore parts of obs we want to leave alone
            obs = (1 - self.ignore_mask) * obs + self.ignore_mask * obs_orig 

            # Scale timestep
            if self.has_timestep:
                obs[:, -1] *= self.time_scale

            return obs
        else:
            return obs

    

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.raw_obs = obs
        return self._obfilt(obs)

