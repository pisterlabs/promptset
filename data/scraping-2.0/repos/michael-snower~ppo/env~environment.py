# analytical tools
import numpy as np

# env
from collections import defaultdict
import gym
from gym.wrappers import FlattenDictWrapper
from env.atari_wrappers import make_atari, wrap_deepmind

class Env(object):

    def __init__(self, env_type, env_num):
        self.game = gym.make(env_type)
        self.env_num = env_num
        self.num_actions = self.game.action_space.n
        self.state_shape = list(self.game.observation_space.shape)
        self.state = self._reset()
        self.done = False
        # recurrent state - only used with recurrent networks, init state is all 0s
        self.rec_state = np.zeros([1, 256], dtype=float)
        # stats
        self.last_game_reward = self.cur_game_reward = 0
        self.game_score = 0

    def step(self, action):
        '''
        Steps to new state in env.
        '''
        if self.done:
            self.done = False
        state, reward, done, info = self.game.step(action)
        self.state = state
        self.done = done
        if done:
            self.state = self._reset()
        return [state, action, reward, done, info]

    def render(self):
        '''
        Renders game.
        '''
        return self.game.render()

    def _reset(self):
        '''
        Resets to new game.
        '''
        return self.game.reset()

class EnvWrapper(object):

    def __init__(self, env, env_num, env_type):
        self.game = env
        self.env_num = env_num
        self.num_actions = self.game.action_space.n
        self.state_shape = list(self.game.observation_space.shape)
        self.state = self._reset()
        self.done = False
        # recurrent state - only used with recurrent networks, init state is all 0s
        self.rec_state = np.zeros([1, 256], dtype=np.float32) # 2nd dim must be lstm_size*2
        # stats
        self.last_game_reward = self.cur_game_reward = 0
        self.game_score = 0

    def step(self, action):
        '''
        Steps to new state in env.
        '''
        if self.done:
            self.done = False
        state, reward, done, info = self.game.step(action)
        self.state = state
        self.done = done
        if done:
            self.state = self._reset()
        return [state, action, reward, done, info]

    def render(self):
        '''
        Renders game.
        '''
        return self.game.render()

    def _reset(self):
        '''
        Resets to new game.
        '''
        return self.game.reset()


# The below code builds environments based on type. Adapted from OpenAI Baselines
# and Deepmind repos. Note that OpenAI Baselines has additional environment
# wrappers for robotics simulations so these may not work well.

def build_env(env_id, env_num):
    env_type = _get_env_type(env_id)
    if env_type == 'atari':
        env = make_atari(env_id)
        if isinstance(env.observation_space, gym.spaces.Dict):
            keys = env.observation_space.spaces.keys()
            env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))
        env = wrap_deepmind(env, frame_stack=4)
        return EnvWrapper(env, env_num, env_type)
    else:
        env = gym.make(env_id)
        return EnvWrapper(env, env_num, env_type)

def _get_env_type(env_id):
    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)