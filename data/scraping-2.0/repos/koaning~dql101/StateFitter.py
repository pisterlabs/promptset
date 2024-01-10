import keras
import numpy as np
import itertools as it
from collections import namedtuple, deque
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD, Adam

class Agent():
    '''
    This object contains methods relevant for the agent.
    '''
    def __init__(self, env, value_mod, epsilon=0.1):
        self.value_mod = value_mod
        self.epsilon = epsilon
        self.env = env

    def action(self, state_now):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.env.action_space.n)
        X = np.matrix(list(state_now))
        return np.argmax(self.value_mod.predict(X))

    def set_episolon(self, eps):
        self.epsilon = eps

class Game():
    '''
    This object keeps track of all environment state. Needs an OpenAI env and an agent.
    '''

    def __init__(self, env, agent, gamma = 0.9, que_len = 200, max_game_len = 200):
        self.env = env
        self.state_que = deque(maxlen = que_len)
        self.episode_scores = []
        self.transition_tup = namedtuple('Transition', 'old_obs, new_obs, action, value')
        self.agent = agent
        self.gamma = gamma
        self.max_game_len = max_game_len

    def run_episode(self, render = False):
        obs_old = self.env.reset()
        data = []
        last_val = 0
        for t in range(self.max_game_len):
            action = self.agent.action(obs_old)
            obs_new, reward, done, info = self.env.step(action)
            if render:
                self.env.render()
            data.append(self.transition_tup(obs_old, obs_new, action, 1))
            obs_old = obs_new
            if done:
                for timestep in range(t)[::-1]:
                    last_val = reward + self.gamma * last_val
                    que_val = self.transition_tup(
                        data[timestep].old_obs,
                         data[timestep].new_obs,
                        data[timestep].action,
                        last_val
                    )
                    self.state_que.append(que_val)
                self.episode_scores.append(t)
                break
        return t

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class ValueModel():
    '''
    This object keeps track of the value model. It needs an environment from OpenAI
    and it can be passed to an agent.
    '''
    def __init__(self, env):
        self.input_size = len(env.state)
        self.output_size = env.action_space.n
        self.models = []
        self.history = LossHistory()
        main_input = Input(shape=(self.input_size,), name = "input")
        mod = Dense(output_dim=10, input_dim=self.input_size , activation="relu")(main_input)
        mod = Dense(output_dim=10, activation="relu")(mod)
        self.base_model = mod
        for i in range(self.output_size):
            pre_output_layer = Dense(output_dim=10, activation="relu")(self.base_model)
            output_node = Dense(output_dim=1, activation="linear")(pre_output_layer)
            output_model = Model(input = main_input, output = output_node)
            output_model.compile(loss='mse', optimizer = Adam(lr=0.001))
            self.models.append(output_model)

    def fit(self, game_hist):
        game_hist = sorted(game_hist, key = lambda x: x.action)
        for action, action_hist in it.groupby(game_hist, lambda _: _.action):
            action_hist = list(action_hist)
            X = np.matrix([list(s.old_obs) for s in action_hist])
            Y = np.matrix([[s.value] for s in action_hist])
            self.models[action].fit(X, Y, nb_epoch=10, callbacks=[self.history], verbose = False)
        return self.history

    def predict(self, state_in):
        return [_.predict(state_in)[0][0] for _ in self.models]