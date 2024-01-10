"""
This module contains the implementation of the Classes: BaseEnvironment, BaseWrapper, BaseObservationWrapper, BaseActionWrapper,
BaseRewardWrapper, BaseGridWorld, BaseCarOnHill, BaseCartPole, BaseInvertedPendulum and LQG.

Then there are the Mujoco Environments Wrappers: BaseMujoco, BaseHalfCheetah, BaseAnt, BaseHopper, BaseHumanoid, BaseSwimmer, 
BaseWalker2d

The Class BaseEnvironment inherits from the Class AbstractUnit and from ABC.

The Class BaseEnvironment is an abstract Class used as base class for all types of environments.

The Class BaseWrapper is used as generic wrapper Class. The Classes BaseObservationWrapper, BaseActionWrapper and 
BaseRewardWrapper are abstract Classes, that when sub-classed can be used to wrap something specific of an environment.

The Classes BaseGridWorld, BaseCarOnHill, BaseCartPole, BaseInvertedPendulum simply wrap the corresponding Classes of MushroomRL
but they are changed so to inherit from the Class BaseEnvironment so that they can be used in this library.

The Class LQG is a re-adaptation of code copied from: https://github.com/T3p/potion/blob/master/potion/envs/lq.py

The Classes BaseHalfCheetah, BaseAnt, BaseHopper, BaseHumanoid, BaseSwimmer and BaseWalker2d inherit from the Class BaseMujoco
and are simply wrappers of the corresponding OpenAI gym Classes.
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy
import math

from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv

from mushroom_rl.utils.spaces import Box
from mushroom_rl.core.environment import MDPInfo
from mushroom_rl.environments import GridWorld, CarOnHill, CartPole, InvertedPendulum

from ARLO.abstract_unit.abstract_unit import AbstractUnit


class BaseEnvironment(AbstractUnit, ABC):
    """
    This is the base environment Class based on the OpenAI Gym class. Part of this class is a re-adaptation of code copied from: 
    -OpenAI gym: 
    cf. https://github.com/openai/gym/blob/master/gym/core.py
    
    -MushroomRL:
    cf. https://github.com/MushroomRL/mushroom-rl/blob/dev/mushroom_rl/core/environment.py
        
    This can be sub-classed by the user to create their own specific environment. 
    
    You must use environment spaces from  MushroomRL (e.g: Box, Discrete).
    
    This Class is an abstract Class and it inherits from the Class AbstractUnit.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):        
        """
        Non-Parameters Members
        ----------------------
        observation_space: This must be a space from MushroomRL like Box, Discrete.
            
        action_space: This must be a space from MushroomRL like Box, Discrete.
       
        gamma: This is the value of the gamma of the MDP, that is in this Class.
        
        horizon: This is the horizon of the MDP, that is in this Class.
        
        The other parameters and non-parameters members are described in the Class AbstractUnit.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)
                    
        self.action_space = None
        self.observation_space = None
        self.gamma = None
        self.horizon = None
    
    def __repr__(self):
         return 'BaseEnvironment('+'observation_space='+str(self.observation_space)+', action_space='+str(self.action_space)\
                +', gamma='+str(self.gamma)+', horizon='+str(self.horizon)+', obj_name='+str(self.obj_name)\
                +', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', logger='+str(self.logger)+')'
                
    @abstractmethod
    def step(self, action):
        """
        Method used to run one step of the environment dynamics.
        """
        
        raise NotImplementedError

    @abstractmethod
    def reset(self, state=None):
        """
        Method used to reset the environment.
        """
        
        raise NotImplementedError

    @abstractmethod
    def render(self, mode='human'):
        """
        Method used to render an environment.
        """
        
        raise NotImplementedError

    def close(self):
        """
        Method used to perform necessary cleanup.
        """
         
        pass

    def seed(self, seed=None):
        """
        Method used to seed the environment.
        """
        
        return
    
    def stop(self):
        """
        Method used to stop an MDP. This is needed for backward compatibility with MushroomRL.
        """
        
        pass
    
    def unwrapped(self):
        """
        This method completely unwraps the env contained in the class.
        
        Returns:
        ----------
        self: The base non-wrapped env instance
        """
        
        return self

    @property
    def info(self):
        """
        Property method that constructs an object of Class mushroom_rl.environment.MDPInfo
        """
        
        #each block must modify the observation_space and action_space according to the transformation they did
        return MDPInfo(observation_space=self.observation_space, action_space=self.action_space, gamma=self.gamma, 
                       horizon=self.horizon)
        
    def _sample_from_box(self, space):
        """
        Parameters
        ----------
        space: The space to which to sample from. It must be an object of Class Box.
        
        This method was copied from OpenAI gym: cf. https://github.com/openai/gym/blob/master/gym/spaces/box.py
        
        Generates a single random sample inside of the Box. In creating a sample of the box, each coordinate is sampled according
        to the form of the interval:
        * [a, b] : uniform distribution
        * [a, inf) : shifted exponential distribution
        * (-inf, b] : shifted negative exponential distribution
        * (-inf, inf) : normal distribution
        """
        
        if(not isinstance(space, Box)):
            exc_msg = 'The method \'_sample_from_box\' can only be applied on \'Box\' spaces!'
            self.logger.exception(msg=exc_msg)
            raise TypeError(exc_msg)
        else:
            bounded_below = -np.inf < space.low
            bounded_above = np.inf > space.high
            
            if(space.high.dtype.kind == 'f'):
                high = space.high  
            else:
                high = space.high.astype('int64') + 1
            
            sample = np.empty(space.shape)
    
            #Masking arrays which classify the coordinates according to interval type
            unbounded = ~bounded_below & ~bounded_above
            upp_bounded = ~bounded_below & bounded_above
            low_bounded = bounded_below & ~bounded_above
            bounded = bounded_below & bounded_above
    
            #Vectorized sampling by interval type
            sample[unbounded] = self.local_prng.normal(size=unbounded[unbounded].shape)
    
            sample[low_bounded] = self.local_prng.exponential(size=low_bounded[low_bounded].shape) + space.low[low_bounded]
            
            sample[upp_bounded] = -self.local_prng.exponential(size=upp_bounded[upp_bounded].shape) + space.high[upp_bounded]
    
            sample[bounded] = self.local_prng.uniform(low=space.low[bounded], high=high[bounded], size=bounded[bounded].shape)
            
            if space.high.dtype.kind == 'i':
                sample = np.floor(sample)
            
            sample = sample.astype(space.high.dtype)
        
            return sample
            
    def sample_from_box_action_space(self):
        """
        This method samples from a Box Action Space.
        """
        
        sample = self._sample_from_box(space=self.action_space)
        
        return sample
                
    def sample_from_box_observation_space(self):
        """
        This method samples from a Box Observation Space.
        """
        
        sample = self._sample_from_box(space=self.observation_space)
        
        return sample
            
    def set_params(self, params_dict):
        """
        Parameters
        ----------
        params_dict: This is a dictionary containing the parameters of the environment and their new value.
        """
        
        if(isinstance(params_dict, dict)):
            for tmp_key in list(params_dict.keys()):
                if(hasattr(self, tmp_key)):
                    setattr(self, tmp_key, params_dict[tmp_key])
                else:
                    exc_msg = 'The environment does not have the member \''+str(tmp_key)+'\'!'
                    self.logger.exception(msg=exc_msg)
                    raise AttributeError(exc_msg)
        else:
            exc_msg = '\'params_dict\' must be a dictionary!'
            self.logger.exception(msg=exc_msg)
            raise TypeError(exc_msg)
      
    def get_params(self, params_names):
        """
        Parameters
        ----------
        params_names: This is a list of strings and they represent the names of the parameters of which we want to get the value.
        
        Returns
        -------
        params_dict: This is a dictionary with as keys the strings in the list params_names and as values the current value of 
                     such parameter.
        """
        
        params_dict = {}
        
        for tmp_key in params_names:
            if(hasattr(self, tmp_key)):
                params_dict.update({tmp_key: getattr(self, tmp_key)})
            else:
                exc_msg = 'The environment does not have the member \''+str(tmp_key)+'\'!'
                self.logger.exception(msg=exc_msg)
                raise AttributeError(exc_msg)
                
        return params_dict
                
        
class BaseWrapper(BaseEnvironment):
    """
    This is the base wrapper Class based on the OpenAI Wrapper Class. Part of this class is a re-adaptation of code copied from 
    OpenAI gym: https://github.com/openai/gym/blob/master/gym/core.py
    
    This is used as base Class for observation wrappers, action wrappers and reward wrappers.
    
    Other kind of wrappers can be created inheriting from this Class.
    """
    
    def __init__(self, env, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):
        """
        Parameters
        ----------
        env: This is the environment that needs to be wrapped. It must be an object of a Class inheriting from the Class 
             BaseEnvironment.
        
        The other parameters and non-parameters members are described in the Class BaseEnvironment.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)

        self.env = env
        
        #The env must be an object of a Class inheriting from the Class BaseEnvironment
        if(not isinstance(self.env, BaseEnvironment)):
            exc_msg = 'The \'env\' must be an object of a Class inheriting from the Class BaseEnvironment!'
            self.logger.exception(msg=exc_msg)
            raise TypeError(exc_msg)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.gamma = self.env.gamma
        self.horizon = self.env.horizon
        
    def __repr__(self):
         return str(self.__class__.__name__)+'('+'env='+str(self.env)+', observation_space='+str(self.observation_space)\
                +', action_space='+str(self.action_space)+', gamma='+str(self.gamma)+', horizon='+str(self.horizon)\
                +', obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)\
                +', log_mode='+str(self.log_mode)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
                +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
                +', logger='+str(self.logger)+')'
        
    def step(self, action):
        """
        Method used to run one step of the environment dynamics.
        """
        
        return self.env.step(action=action)

    def reset(self, state=None): 
        """
        Method used to reset the environment.
        """
        
        return self.env.reset(state=state)

    def render(self, mode='human'):   
        """
        Method used to render an environment.
        """
        
        return self.env.render(mode=mode)

    def close(self):  
        """
        Method used to perform necessary cleanup.
        """
         
        return self.env.close()

    def seed(self, seed=None):
        """
        Method used to seed the environment.
        """
        
        return self.env.seed(seed=seed)
   
    def stop(self):
        return self.env.stop()

    def unwrapped(self):
        """
        Method to unwrap the environment.
        """
        
        return self.env.unwrapped()
    
    
class BaseObservationWrapper(BaseWrapper):
    """
    Part of this Class is a readaptation of code copied from OpenAI gym: https://github.com/openai/gym/blob/master/gym/core.py
    
    To create an observation wrapper you must create a new Class inheriting from this Class. You must override the observation 
    method, and in the __init__ you must:
        -Call the __init__ of BaseWrapper via: super().__init__(env)
        -Properly modify the observation space.
        
    This is an abstract Class: it must be sub-classed.
    """
    
    def reset(self, state=None):
        """
        Method used to reset the environment.
        """
        
        observation = self.env.reset(state=state)
        return self.observation(observation=observation)

    def step(self, action):
        """
        Method used to run one step of the environment dynamics.
        """
        
        observation, reward, absorbing, info = self.env.step(action=action)
        return self.observation(observation=observation), reward, absorbing, {}

    @abstractmethod
    def observation(self, observation):
        """
        Method used to transform the observations.
        """
        
        raise NotImplementedError    


class BaseActionWrapper(BaseWrapper):
    """
    Part of this Class is a readaptation of code copied from OpenAI gym: https://github.com/openai/gym/blob/master/gym/core.py
    
    To create an action wrapper you must create a new Class inheriting from this Class. You must override the action method and 
    in the __init__ you must:
        -Call the __init__ of BaseWrapper via: super().__init__(env)
        -Properly modify the action space.
    
    This is an abstract Class: it must be sub-classed.
    """

    def step(self, action):
        """
        Method used to run one step of the environment dynamics.
        """
        
        return self.env.step(self.action(action=action))

    @abstractmethod
    def action(self, action):
        """
        Method used to transform the actions.
        """
        
        raise NotImplementedError
        

class BaseRewardWrapper(BaseWrapper):
    """
    Part of this Class is a readaptation of code copied from OpenAI gym: https://github.com/openai/gym/blob/master/gym/core.py
    
    To create a reward wrapper you must create a new Class inheriting from this Class. You must override the reward method, and
    in the __init__ you must:
        -Call the __init__ of BaseWrapper via: super().__init__(env)   
    
    This is an abstract Class: it must be sub-classed.
    """
    
    def step(self, action):
        """
        Method used to run one step of the environment dynamics.
        """
        
        observation, reward, absorbing, info = self.env.step(action=action)
        return observation, self.reward(reward=reward), absorbing, {}

    @abstractmethod
    def reward(self, reward):
        """
        Method used to transform the rewards.
        """
        
        raise NotImplementedError
    
    
class BaseGridWorld(GridWorld, BaseEnvironment):
    """
    This Class wraps the GridWorld Class from MushroomRL: this is needed for the correct working of this library.
    """
    
    def __init__(self, height, width, goal, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, 
                 n_jobs=1, job_type='process', start=(0,0)):
        super().__init__(height=height, width=width, goal=goal, start=start)
        super(BaseEnvironment, self).__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                                              checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, 
                                              job_type=job_type)  
        self.horizon = super().info.horizon
        self.gamma = super().info.gamma
        
        self.observation_space   = super().info.observation_space
        self.action_space =  super().info.action_space
            
    def __repr__(self):
        return 'BaseGridWorld('+'observation_space='+str(self.observation_space)+', action_space='+str(self.action_space)\
                +', gamma='+str(self.gamma)+', horizon='+str(self.horizon)+', obj_name='+str(self.obj_name)\
                +', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', logger='+str(self.logger)+')'

    def seed(self, seed=None):
        """
        Method used to seed the environment.
        """
        
        return
   
    
class BaseCarOnHill(CarOnHill, BaseEnvironment):
    """
    This Class wraps the CarOnHill Class from MushroomRL: this is needed for the correct working of this library.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process', horizon=100, gamma=.95):
        super().__init__(horizon=horizon, gamma=gamma)
        super(BaseEnvironment, self).__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                                              checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs, 
                                              job_type=job_type)  
      
        self.horizon = super().info.horizon
        self.gamma = super().info.gamma
      
        self.observation_space = super().info.observation_space
        self.action_space =  super().info.action_space
                    
    def __repr__(self):
        return 'BaseCarOnHill('+'observation_space='+str(self.observation_space)+', action_space='+str(self.action_space)\
                +', gamma='+str(self.gamma)+', horizon='+str(self.horizon)+', obj_name='+str(self.obj_name)\
                +', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', logger='+str(self.logger)+')'   
      
    def seed(self, seed=None):
        """
        Method used to seed the environment.
        """
        
        return
   
    
class BaseCartPole(CartPole, BaseEnvironment):
    """
    This Class wraps the CartPole Class from MushroomRL: this is needed for the correct working of this library.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process', m=2., M=8., l=.5, g=9.8, mu=1e-2, max_u=50., noise_u=10., horizon=3000, gamma=.95):
        super().__init__(m=m, M=M, l=l, g=g, mu=mu, max_u=max_u, noise_u=noise_u, horizon=horizon, gamma=gamma)
        super(BaseEnvironment, self).__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                                              checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs,
                                              job_type=job_type)   
      
        self.horizon = super().info.horizon
        self.gamma = super().info.gamma
      
        self.observation_space  = super().info.observation_space
        self.action_space  =  super().info.action_space
      
    def __repr__(self):
        return 'BaseCartPole('+'observation_space='+str(self.observation_space)+', action_space='+str(self.action_space)\
                +', gamma='+str(self.gamma)+', horizon='+str(self.horizon)+', obj_name='+str(self.obj_name)\
                +', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)+', log_mode='+str(self.log_mode)\
                +', checkpoint_log_path='+str(self.checkpoint_log_path)+', verbosity='+str(self.verbosity)\
                +', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)+', logger='+str(self.logger)+')'  
                
    def seed(self, seed=None):
        """
        Method used to seed the environment.
        """
        
        return          
   
    
class BaseInvertedPendulum(InvertedPendulum, BaseEnvironment):
    """
    This Class wraps the InvertedPendulum Class from MushroomRL: this is needed for the correct working of this library.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process', random_start=False, m=1., l=1., g=9.8, mu=1e-2, max_u=5., horizon=5000, gamma=.99):
        super().__init__(random_start=random_start, m=m, l=l, g=g, mu=mu, max_u=max_u, horizon=horizon, gamma=gamma)
        super(BaseEnvironment, self).__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, 
                                              checkpoint_log_path=checkpoint_log_path, verbosity=verbosity, n_jobs=n_jobs,
                                              job_type=job_type)   
        
        self.horizon = super().info.horizon
        self.gamma = super().info.gamma
      
        self.observation_space = super().info.observation_space
        self.action_space =  super().info.action_space
            
    def __repr__(self):
        return 'BaseInvertedPendulum('+'observation_space='+str(self.observation_space)\
                +', action_space='+str(self.action_space)+', gamma='+str(self.gamma)+', horizon='+str(self.horizon)\
                +', obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)\
                +', log_mode='+str(self.log_mode)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
                +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
                +', logger='+str(self.logger)+')'  
                 
    def seed(self, seed=None):
        """
        Method used to seed the environment.
        """
        
        return
   
    
class LQG(BaseEnvironment):
    """
    Environment implementing a Linear-Quadratic Gaussian control (LQG) problem: s_{t+1} = A s_t + B a_t + env_noise
    
    The reward function is given by: r_{t+1} = - s_t^T Q s_t - a_t^T R a_t
    
    Note that there is also a noise on the controller: if you pick action a_t then you will be able to execute action: 
    a_t + controller_noise. This only plays a role when rolling out the policy: we consider a gaussian policy with mean a_t and
    covariante matrix given by controller_noise.
    
    This is a re-adaptation of code copied from: https://github.com/T3p/potion/blob/master/potion/envs/lq.py
    """ 

    def __init__(self, obj_name, A=np.eye(1), B=np.eye(1), Q=np.eye(1), R=np.eye(1), max_pos=1.0, max_action=1.0, 
                 env_noise=np.eye(1), controller_noise=np.eye(1), horizon=10, gamma=0.9, seeder=2, log_mode='console', 
                 checkpoint_log_path=None, verbosity=3, n_jobs=1, job_type='process'):
        """
        Parameters
        ----------
        A: This is the state dynamics matrix.
        
           The default is np.eye(1).
       
        B: This is the action dynamics matrix.
        
           The default is np.eye(1).
        
        Q: This is the cost weight matrix for the state. It must be a positive-definite matrix (to always have a negative reward).
            
           The default is np.eye(1).
        
        R: This is the cost weight matrix for the action. It must be a positive-definite matrix (to always have a negative 
           reward).
            
           The default is np.eye(1).
        
        max_pos: This is the maximum value that the state can reach.
            
                 The default is 1.0.
        
        max_action: This is the maximum value that the action can reach.
        
                    The default is 1.0.
        
        env_noise: This is the covariance matrix representing the environment noise.
                    
                   The default is np.eye(1).
      
        controller_noise: This is the covariance matrix representing the controller noise.
                    
                          The default is np.eye(1).
                   
        horizon: This is the horizon of the MDP. 
                
                 The default is 10.
            
        gamma: This is the discount factor of the MDP.
            
               The default is 0.9.
        
        Non-Parameters Members
        ----------------------
        is_eval_phase: This is True if the environment is used for evaluating a policy: what happens is that the controller_noise
                       is added to the action selected by the policy, and then fed to the simulator. 
                      
                       Otherwise it is False.
                      
                       This is used to represent the fact that even if we have learnt a theoretically optimal policy, in practice 
                       to execute it there is going to be some noise and so the resulting action taken in the real world will be
                       different from the one selected by the policy.
                      
                       This parameter can be set automatically by the evaluation metric.
        
        The other parameters and non-parameters members are described in the Class BaseEnvironment.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity, n_jobs=n_jobs, job_type=job_type)   
        
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
        #state dimension
        self.ds = self.A.shape[1]
        
        #action dimension
        self.da = self.B.shape[1] 
        
        if(self.da == 1):
            if(self.R < 0):
                exc_msg = 'The matrix \'R\' must be positive-definite so that the reward is always negative!'
                self.logger.exception(msg=exc_msg)
                raise ValueError(exc_msg)
        else:
            if(not np.all(np.linalg.eigvals(self.R) > 0)):
                exc_msg = 'The matrix \'R\' must be positive-definite so that the reward is always negative!'
                self.logger.exception(msg=exc_msg)
                raise ValueError(exc_msg)
            
        if(self.ds == 1):
            if(self.Q < 0):
                exc_msg = 'The matrix \'Q\' must be positive-definite so that the reward is always negative!'
                self.logger.exception(msg=exc_msg)
                raise ValueError(exc_msg)
        else:
            if(not np.all(np.linalg.eigvals(self.Q) > 0)):
                exc_msg = 'The matrix \'Q\' must be positive-definite so that the reward is always negative!'
                self.logger.exception(msg=exc_msg)
                raise ValueError(exc_msg)
        
        #task horizon 
        self.horizon = horizon 
        
        #discount factor
        self.gamma = gamma
        
        #max state for clipping
        self.max_pos = max_pos*np.ones(self.ds) 
        
        #max action for clipping 
        self.max_action = max_action*np.ones(self.da) 
        
        #environment noise
        self.env_noise = env_noise
        #check that the env_noise has the right shape:
        if(self.env_noise.shape[1] != self.ds):
            exc_msg = 'The \'env_noise\' co-variance matrix is not of the right shape!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)
            
        self.viewer = None

        self.action_space = Box(low=-self.max_action, high=self.max_action)
                                      
        self.observation_space = Box(low=-self.max_pos, high=self.max_pos)
        
        self.controller_noise = controller_noise
        #check that the controller_noise has the right shape:
        if(self.controller_noise.shape[1] != self.da):
            exc_msg = 'The \'controller_noise\' co-variance matrix is not of the right shape!'
            self.logger.exception(msg=exc_msg)
            raise ValueError(exc_msg)            
 
        #when this is true the controller noise is added to the action that the policy originally picked.
        self.is_eval_phase = False
                
    def __repr__(self):
        return 'LQG('+'observation_space='+str(self.observation_space)+', action_space='+str(self.action_space)\
                +', gamma='+str(self.gamma)+', horizon='+str(self.horizon)+', A='+str(self.A)+', B='+str(self.B)\
                +', Q='+str(self.Q)+', R='+str(self.R)+', env_noise='+str(self.env_noise)\
                +', controller_noise='+str(self.controller_noise)+', is_eval_phase='+str(self.is_eval_phase)\
                +', obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)\
                +', log_mode='+str(self.log_mode)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
                +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
                +', logger='+str(self.logger)+')'   

    def step(self, action):
        """
        Method used to run one step of the environment dynamics.
        """
        
        if(self.is_eval_phase):
            action = action + self.local_prng.multivariate_normal(mean=np.zeros(self.da), cov=self.controller_noise)
        
        u = np.clip(np.ravel(np.atleast_1d(action)), -self.max_action, self.max_action)
        
        #makes the noise different at each step:
        env_noise = np.dot(self.env_noise, self.local_prng.standard_normal(self.ds))
                
        xn = np.clip(np.dot(self.A, self.state.T) + np.dot(self.B, u) + env_noise, -self.max_pos, self.max_pos)
        
        cost = np.dot(self.state, np.dot(self.Q, self.state)) + np.dot(u, np.dot(self.R, u))

        self.state = xn.ravel()
        
        self.timestep += 1
                
        return np.array(self.state), -np.asscalar(cost), self.timestep >= self.horizon, {'danger':0} 

    def reset(self, state=None):
        """
        By default, random uniform initialization. 
        """
        
        self.timestep = 0
        if state is None:
            self.state = np.array(self.local_prng.uniform(low=-self.max_pos, high=self.max_pos, size=self.ds))
        else:
            self.state = np.array(state)

        return np.array(self.state)

    def seed(self, seed=None):
        """
        Method used to seed the environment.
        """
        
        if(seed is None):        
            self.set_local_prng(new_seeder=seed)
        
    def render(self, mode='human', close=False):
        """
        Method used to render the environment.
        """
        
        #this is here since it would otherwise open a plot window
        from gym.envs.classic_control import rendering

        if self.ds not in [1, 2]:
            return
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        world_width = math.ceil((self.max_pos[0] * 2) * 1.5)
        xscale = screen_width / world_width
        ballradius = 3
        
        if self.ds == 1:    
            screen_height = 400
        else:
            world_height = math.ceil((self.max_pos[1] * 2) * 1.5)
            screen_height = math.ceil(xscale * world_height)
            yscale = screen_height / world_height

        if self.viewer is None:
            clearance = 0  # y-offset
            self.viewer = rendering.Viewer(screen_width, screen_height)
            mass = rendering.make_circle(ballradius * 2)
            mass.set_color(.8, .3, .3)
            mass.add_attr(rendering.Transform(translation=(0, clearance)))
            self.masstrans = rendering.Transform()
            mass.add_attr(self.masstrans)
            self.viewer.add_geom(mass)
            if self.ds == 1:
                self.track = rendering.Line((0, 100), (screen_width, 100))
            else:
                self.track = rendering.Line((0, screen_height / 2), (screen_width, screen_height / 2))
            self.track.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(self.track)
            zero_line = rendering.Line((screen_width / 2, 0), (screen_width / 2, screen_height))
            zero_line.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(zero_line)

        x = self.state[0]
        ballx = x * xscale + screen_width / 2.0
        if self.ds == 1:
            bally = 100
        else:
            y = self.state[1]
            bally = y * yscale + screen_height / 2.0
        self.masstrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        
    def get_optimal_K(self):
        """
        Returns
        -------
        Computes the optimal parameters K and returns the optimal policy given by: -K*s where s is the state.
        
        Note that the addition of the controller noise is taken care of in the evaluation phase.
        """ 
        
        X = np.matrix(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))

        K = np.matrix(scipy.linalg.inv(self.B.T*X*self.B+self.R)*(self.B.T*X*self.A))
                
        return -K
    
    
class BaseMujoco(BaseEnvironment):
    """
    This Class wraps the Mujoco environments. Every Mujoco environment inherits from this Class. 
    
    This is an abstract Class.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process'):                 
        """
        Non-Parameters Members
        ----------------------
        mujoco_env: This is a Mujoco environment instance: an object of a Class inheriting from OpenAI gym Class: 
                    gym.envs.mujoco.mujoco_env.MujocoEnv.
            
        n_steps: This is a counter for the horizon of the environment. It is the number of total steps that have happened so far.
        """    
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity)
        
        #this is set in the sub-Class:
        self.mujoco_env = None
        
        self.n_steps = 0
                                
    def __repr__(self):
        return str(self.__class__.__name__)+'('+'observation_space='+str(self.observation_space)\
               +', action_space='+str(self.action_space)+', gamma='+str(self.gamma)+', horizon='+str(self.horizon)\
               +', obj_name='+str(self.obj_name)+', seeder='+str(self.seeder)+', local_prng='+str(self.local_prng)\
               +', log_mode='+str(self.log_mode)+', checkpoint_log_path='+str(self.checkpoint_log_path)\
               +', verbosity='+str(self.verbosity)+', n_jobs='+str(self.n_jobs)+', job_type='+str(self.job_type)\
               +', logger='+str(self.logger)+')'  
                               
    def _update_counter(self, out_step):
        """
        Parameters
        ----------
        out_step: The output from the mujoco_env object step() method.
        
        Returns 
        -------
        tuple(out_step): The modified environment step() method output.
        
        Updates the n_steps and if the horizon is reached the output of the method step() from the environment is modified: done
        is set to True.
        """
        
        if(hasattr(self, 'n_steps')):
            out_step = list(out_step)
            
            self.n_steps += 1
        
            if(self.n_steps >= self.horizon):
                out_step[2] = True
                self.n_steps = 0
            
        return tuple(out_step)
  
    def step(self, action):
        """
        Method used to run one step of the environment dynamics.
        """
        
        out = self.mujoco_env.step(action=action)
        
        out = self._update_counter(out_step=out)
        
        return out 
           
    def reset(self, state=None):
        """
        Method used to reset the environment.
        """
        
        obs = self.mujoco_env.reset()
        
        self.n_steps = 0
        
        return obs
    
    def seed(self, seed=None):
        """
        Method used to seed the environment.
        """
        
        return self.mujoco_env.seed(seed=seed)
        
    def render(self, mode='human'):
        """
        Method used to render the environment.
        """
        
        return self.mujoco_env.render(mode=mode)
    
    def set_local_prng(self, new_seeder):
        """
        Method used to adjust to the fact that OpenAI does not use my system of using local prng but have their own way of doing
        it. Without this method by calling the original method set_local_prng() implemented in the Class AbstractUnit there would
        be no effect on the environment.
        
        The method set_local_prng() is called in the Metric Classes and in some of the DataGeneration Classes so this method 
        needs to work properly.
        """
        
        self.seed(seed=new_seeder)
    
    def _initialise_mdp_properties(self, gamma, horizon):
        """
        Parameters
        ----------
        gamma: This is the MDP discount factor. It must be a float.
        
        horizon: This is the MDP horizon. It must be an integer.
        """
        
        #no gamma, no horizon in Mujoco: i select them
        self.gamma = gamma
        self.horizon = horizon
        
        #I need to use the environment spaces from MushroomRL:
        self.action_space = Box(self.mujoco_env.action_space.low, self.mujoco_env.action_space.high)
        
        self.observation_space = Box(self.mujoco_env.observation_space.low, self.mujoco_env.observation_space.high)
    
    
class BaseHalfCheetah(BaseMujoco):
    """
    This Class wraps the Mujoco environment: HalfCheetahEnv.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process', gamma=0.99, horizon=1000, xml_file='half_cheetah.xml', forward_reward_weight=1.0, 
                 ctrl_cost_weight=0.1, reset_noise_scale=0.1, exclude_current_positions_from_observation=True):
        """
        Parameters
        ----------
        gamma: This is the MDP discount factor. It must be a float.
         
               The default is 0.99.
        
        horizon: This is the MDP horizon. It must be an integer.
            
                 The default is 1000.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity)
        
        self.mujoco_env = HalfCheetahEnv(xml_file=xml_file, forward_reward_weight=forward_reward_weight, 
                                         ctrl_cost_weight=ctrl_cost_weight, reset_noise_scale=reset_noise_scale,
                                         exclude_current_positions_from_observation=exclude_current_positions_from_observation)
        
        self._initialise_mdp_properties(gamma=gamma, horizon=horizon)

        
class BaseAnt(BaseMujoco):
    """
    This Class wraps the Mujoco environment: AntEnv.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process', gamma=0.99, horizon=1000, xml_file='ant.xml', ctrl_cost_weight=0.5, contact_cost_weight=5e-4, 
                 healthy_reward=1.0, terminate_when_unhealthy=True, healthy_z_range=(0.2, 1.0), contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1, exclude_current_positions_from_observation=True):
        """
        Parameters
        ----------
        gamma: This is the MDP discount factor. It must be a float.
         
               The default is 0.99.
        
        horizon: This is the MDP horizon. It must be an integer.
            
                 The default is 1000.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity)
    
        self.mujoco_env = AntEnv(xml_file=xml_file, ctrl_cost_weight=ctrl_cost_weight, contact_cost_weight=contact_cost_weight,
                                 healthy_reward=healthy_reward, terminate_when_unhealthy=terminate_when_unhealthy, 
                                 healthy_z_range=healthy_z_range, contact_force_range=contact_force_range, 
                                 reset_noise_scale=reset_noise_scale, 
                                 exclude_current_positions_from_observation=exclude_current_positions_from_observation)
                
        self._initialise_mdp_properties(gamma=gamma, horizon=horizon)                


class BaseHopper(BaseMujoco):
    """
    This Class wraps the Mujoco environment: HopperEnv.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process', gamma=0.99, horizon=1000, xml_file='hopper.xml', forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-3, healthy_reward=1.0, terminate_when_unhealthy=True, healthy_state_range=(-100.0, 100.0), 
                 healthy_z_range=(0.7, float('inf')), healthy_angle_range=(-0.2, 0.2), reset_noise_scale=5e-3, 
                 exclude_current_positions_from_observation=True):
        """
        Parameters
        ----------
        gamma: This is the MDP discount factor. It must be a float.
         
               The default is 0.99.
        
        horizon: This is the MDP horizon. It must be an integer.
            
                 The default is 1000.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity)
     
        self.mujoco_env = HopperEnv(xml_file=xml_file, forward_reward_weight=forward_reward_weight, 
                                    ctrl_cost_weight=ctrl_cost_weight, healthy_reward=healthy_reward,
                                    terminate_when_unhealthy=terminate_when_unhealthy, healthy_state_range=healthy_state_range, 
                                    healthy_z_range=healthy_z_range, healthy_angle_range=healthy_angle_range, 
                                    reset_noise_scale=reset_noise_scale, 
                                    exclude_current_positions_from_observation=exclude_current_positions_from_observation)
        
        self._initialise_mdp_properties(gamma=gamma, horizon=horizon)
                        
        
class BaseHumanoid(BaseMujoco):
    """
    This Class wraps the Mujoco environment: HumanoidEnv.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process', gamma=0.99, horizon=1000, xml_file='humanoid.xml', forward_reward_weight=1.25, 
                 ctrl_cost_weight=0.1, contact_cost_weight=5e-7, contact_cost_range=(-np.inf, 10.0), healthy_reward=5.0,
                 terminate_when_unhealthy=True, healthy_z_range=(1.0, 2.0), reset_noise_scale=1e-2, 
                 exclude_current_positions_from_observation=True):
        """
        Parameters
        ----------
        gamma: This is the MDP discount factor. It must be a float.
         
               The default is 0.99.
        
        horizon: This is the MDP horizon. It must be an integer.
            
                 The default is 1000.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity)
        
        self.mujoco_env = HumanoidEnv(xml_file=xml_file, forward_reward_weight=forward_reward_weight, 
                                      ctrl_cost_weight=ctrl_cost_weight, contact_cost_weight=contact_cost_weight, 
                                      contact_cost_range=contact_cost_range, healthy_reward=healthy_reward, 
                                      terminate_when_unhealthy=terminate_when_unhealthy, healthy_z_range=healthy_z_range, 
                                      reset_noise_scale=reset_noise_scale, 
                                      exclude_current_positions_from_observation=exclude_current_positions_from_observation)
        
        self._initialise_mdp_properties(gamma=gamma, horizon=horizon)
        
        
class BaseSwimmer(BaseMujoco):
    """
    This Class wraps the Mujoco environment: SwimmerEnv.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process', gamma=0.99, horizon=1000, xml_file="swimmer.xml", forward_reward_weight=1.0, 
                 ctrl_cost_weight=1e-4, reset_noise_scale=0.1, exclude_current_positions_from_observation=True):
        """
        Parameters
        ----------
        gamma: This is the MDP discount factor. It must be a float.
         
               The default is 0.99.
        
        horizon: This is the MDP horizon. It must be an integer.
            
                 The default is 1000.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity)
    
        self.mujoco_env = SwimmerEnv(xml_file=xml_file, forward_reward_weight=forward_reward_weight, 
                                     ctrl_cost_weight=ctrl_cost_weight, reset_noise_scale=reset_noise_scale, 
                                     exclude_current_positions_from_observation=exclude_current_positions_from_observation)
                        
        self._initialise_mdp_properties(gamma=gamma, horizon=horizon)


class BaseWalker2d(BaseMujoco):
    """
    This Class wraps the Mujoco environment: Walker2dEnv.
    """
    
    def __init__(self, obj_name, seeder=2, log_mode='console', checkpoint_log_path=None, verbosity=3, n_jobs=1, 
                 job_type='process', gamma=0.99, horizon=1000, xml_file="walker2d.xml", forward_reward_weight=1.0, 
                 ctrl_cost_weight=1e-3, healthy_reward=1.0, terminate_when_unhealthy=True, healthy_z_range=(0.8, 2.0), 
                 healthy_angle_range=(-1.0, 1.0), reset_noise_scale=5e-3, exclude_current_positions_from_observation=True):
        """
        Parameters
        ----------
        gamma: This is the MDP discount factor. It must be a float.
         
               The default is 0.99.
        
        horizon: This is the MDP horizon. It must be an integer.
            
                 The default is 1000.
        """
        
        super().__init__(obj_name=obj_name, seeder=seeder, log_mode=log_mode, checkpoint_log_path=checkpoint_log_path, 
                         verbosity=verbosity)
        
        self.mujoco_env = Walker2dEnv(xml_file=xml_file, forward_reward_weight=forward_reward_weight, 
                                      ctrl_cost_weight=ctrl_cost_weight, healthy_reward=healthy_reward, 
                                      terminate_when_unhealthy=terminate_when_unhealthy, healthy_z_range=healthy_z_range,
                                      healthy_angle_range=healthy_angle_range, reset_noise_scale=reset_noise_scale, 
                                      exclude_current_positions_from_observation=exclude_current_positions_from_observation)
    
        self._initialise_mdp_properties(gamma=gamma, horizon=horizon)