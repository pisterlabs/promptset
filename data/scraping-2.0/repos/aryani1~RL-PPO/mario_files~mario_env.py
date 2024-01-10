import gym
import numpy as np

from baselines.common.atari_wrappers import FrameStack

class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
    
    def observation(self, frame):
        if frame is None:
            frame = np.zeros((13,16)) # tiles x,y shape
        return frame


class ActionsDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        # From openai github:
        #   Don't forget to call super(class_name, self).init(env) 
        #   if you override the wrapper's init function.
        super(ActionsDiscretizer, self).__init__(env)

        self._actions = np.array([
           [0, 0, 0, 0, 0, 0], #0 - no button",
           [1, 0, 0, 0, 0, 0], #1 - up only (to climb vine)",
           #[0, 0, 1, 0, 0, 0], #2 - left only",
           [0, 0, 0, 1, 0, 0], #3 - right only",
           [0, 0, 0, 0, 0, 1], #4 - run only",
           [0, 0, 0, 0, 1, 0], #5 - jump only",
           #[0, 0, 1, 0, 0, 1], #6 - left run",
           #[0, 0, 1, 0, 1, 0], #7 - left jump",
           [0, 0, 0, 1, 0, 1], #8 - right run",
           [0, 0, 0, 1, 1, 0], #9 - right jump",
           #[0, 0, 1, 0, 1, 1], #10 - left run jump",
           [0, 0, 0, 1, 1, 1]]) #11 - right run jump",

        self.action_space = gym.spaces.Discrete(len(self._actions))

    # take an action
    def action(self, a):
        return self._actions[a].copy()

    # def reset(self):
    #     #print(self.env.change_level(new_level=0))
    #     #return self.env.reset()
    #     return self.env.change_level(new_level=0)


class ProcessRewards(gym.Wrapper):
    def __init__(self, env):
        super(ProcessRewards, self).__init__(env)
        self._max_x  = 41
        self._time_  = 400
        self._score_ = 0
    
    def reset(self, **kwargs):
        # TODO: Try to changelevel to level 0 instead
        #       of reseting the entire environment.
        #       this is would yield faster training.
        self._max_x  = 41
        self._time_  = 400
        self._score_ = 0

        return self.env.reset(**kwargs)
        #return self.env.change_level(new_level=0)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        score_coef   = 0.0001 # tune the score reward
        time_penalty = 0.01 # for every second that passes, give -'time_penalty' reward

        r = 0
        # Check first if distance is in info, this is mario-specific
        if 'distance' in info:
            if info['distance'] > 41:
                r += reward * 0.5

            score_dif = (info['score'] - self._score_) * score_coef
            r += score_dif

            # time penalty every second
            if info['time'] < self._time_:
                r -= time_penalty

            # if mario died
            if done and info['life'] == 0:
                r -= 2
            
            if done and info['distance'] > 0.97 * 3266: # level 0 max_distance
                r += 2

            self._max_x  = max(self._max_x, info['distance'])
            self._score_ = info['score']
            self._time_  = info['time']
        return obs, r, done, info

def replace_nans(obs):
    obs[np.isnan(obs)] = 0.
    return obs

def make_env():
    ''' function for editing and returning the environment for mario '''
    env = gym.make('SuperMarioBros-1-1-v0')
    env = ActionsDiscretizer(env)
    env = ProcessRewards(env)
    env = ObsWrapper(env)
    env.close()
    #env = FrameStack(env, 2)
    return env