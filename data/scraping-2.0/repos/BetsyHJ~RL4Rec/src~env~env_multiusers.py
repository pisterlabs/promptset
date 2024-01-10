'''
 Author: Jin
 Tensorflow
 Build based on Gym.env
'''

# import gym
import numpy as np
import scipy.sparse as sp

class SOFAEnv(object):
    """
    Following the setting from openai gym, we have API methods
    step
    reset
    render? not sure
    close
    seed

    And we have attributes:
    action_space,
    obvervation_space? is this state?
    reward_range ??

    Consider dealing with all users
    """
    def __init__(self, config):
        super(SOFAEnv, self).__init__()
        self.config = config
        # config setting
        self.episode_length = int(self.config['EPISODE_LENGTH'])

        # load ratings file
        self.ratings = self.config['RATINGS']
        if self.config['RATING_TYPE'] == 'matrix':
            (self.num_users, self.num_items) = self.ratings.shape
        else:
            print("wrong rating file type")
            exit(1)
        
        # train and test split (to-do-list)
        self.action_space = np.arange(self.num_items)
        # if one item is not avalible for a user, just mask this slot, self.user_action_mask[u, j] = True
        
        # initialize the env
        self.click_probability()
        self.state = None # [items, feedbacks]
        # np.random.seed(2020)
        self.with_userinit = False
        print("sofa with user_init:", self.with_userinit)

    def reset(self, user_ids):
        self.user_ids = user_ids # to-do-list: multiple users
        self.step_count = 0
        # now state is history_items, to-do-list: user_id/info
        self.state = np.hstack((np.full((len(self.user_ids), self.episode_length), self.num_items, dtype=np.int32), \
                np.full((len(self.user_ids), self.episode_length), 0, dtype=np.int32))).reshape(-1, 2, self.episode_length)
        if self.with_userinit:
            # print(self.state.shape)
            self.state[:, 0, 0] = self.num_items + 1
            self.state[:, 1, 1] = 2

    
    def step(self, actions):
        '''
        Input: Action (item_id or item_ids)
        Return: state, reward, done, {some info}
        '''
        # if action in self.state[0]: # to-do-list: consider to change it into unclick
        # # report error when repeat
        # if np.isin(0, self.state[:, 0] - np.expand_dims(np.array(actions, dtype=np.int32), axis=-1)):
        #     print(actions)
        #     print("recommend repeated item")
        #     exit(0)

        click_flag = self.get_responds(actions)
        # # if repeat item, click_flag is 0
        if np.isin(0, self.state[:, 0] - np.expand_dims(np.array(actions, dtype=np.int32), axis=-1)):
            click_flag[np.where((self.state[:, 0] - np.expand_dims(np.array(actions, dtype=np.int32), axis=-1))==0)[0]] = 0
        # if click_flag == 1:
        #     self.state.append(action)
        self.state[:, 0, self.step_count] = actions
        self.state[:, 1, self.step_count] = click_flag
        self.step_count += 1
        done = False
        if self.step_count >= self.episode_length:
            done = True
            # self.state[:, 0, :] = self.num_items

        return (self.state, click_flag, done)
            

    def click_probability(self):
        # self.click_p = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
        self.click_p = np.array([0, 0, 0, 0, 1., 1.], dtype=np.float32)
    def get_ratings(self, item_ids):
        return self.ratings[self.user_ids, item_ids]
    def get_responds(self, item_ids):
        # click_p = self.get_ratings(item_ids) * 0.2
        # clicks = np.less(np.random.uniform(size=(len(item_ids))), click_p)
        r = self.get_ratings(item_ids)
        clicks = np.less(np.random.uniform(size=(len(item_ids))), self.click_p[r])
        return np.where(clicks, 1, 0)
        # return self.get_ratings(item_ids).astype('int')

# def simulated_data(num_users=500, num_items=200):
#     ratings = np.random.randint(1,6, size=(num_users, num_items))
#     return ratings

def simulated_data(num_users=500, num_items=200):
    import os, pickle
    module_path = os.path.dirname(__file__)  
    if module_path == '':
        module_path = '.'
    filename = module_path + "/simulated_U" + str(num_users) + "_I_" + str(num_items) + '.pkl'
    print(filename)
    if os.path.exists(filename):
        print("load from %s" % filename)
        (ratingM, item_vec) = pickle.load(open(filename, 'rb'))
        unique, counts = np.unique(ratingM, return_counts=True)
        print("the rating count of the simulator, \n", unique, '\n', counts, flush=True)
        return ratingM, item_vec

    n_dim = 5
    # np.random.seed(2020)
    user_vec = np.random.randn(num_users, n_dim)
    item_vec = np.random.randn(num_items, n_dim)
    def sigmoid(x):
        return (1.0 / (1.0 + np.exp(-x)))
    rating_matrix = np.clip(np.round(sigmoid(np.dot(user_vec, item_vec.T) *0.8-1. )*4)+1, 1, 5)
    # count
    unique, counts = np.unique(rating_matrix, return_counts=True)
    print("the rating count of the simulator, \n", unique, '\n', counts, flush=True)
    pickle.dump((np.array(rating_matrix, dtype=np.int32), item_vec), open(filename, 'wb'))
    print("save into %s" % filename)
    return np.array(rating_matrix, dtype=np.int32), item_vec


if __name__ == "__main__":
    import time
    begining_time = time.time()
    simulated_data(10, 20)
    # exit(0)
    config = dict()
    config['RATINGS'], _ = simulated_data(10, 20)
    config['RATING_TYPE'] = 'matrix'
    config['EPISODE_LENGTH'] = 10

    sofa = SOFAEnv(config)
    action_space = sofa.num_items
    users = np.arange(sofa.num_users, dtype=np.int32)
    start_uid = 0
    user_batch = 100
    while start_uid < max(users):
        # initial
        sofa.reset(users[start_uid:(start_uid+user_batch)])
        while True:
            actions = np.random.randint(action_space, size=(len(sofa.user_ids)))
            repeat_actions = np.where((sofa.state[:, 0] - np.expand_dims(np.array(actions, dtype=np.int32), axis=-1) == 0))
            while len(repeat_actions[0]) > 0:
                actions_replace = np.random.randint(action_space, size=(len(sofa.user_ids)))
                actions[repeat_actions[0]] = actions_replace[repeat_actions[0]]
                repeat_actions = np.where((sofa.state[:, 0] - np.expand_dims(np.array(actions, dtype=np.int32), axis=-1) == 0))
            (state, reward, done) = sofa.step(actions)
            if done:
                print("\n", sofa.user_ids)
                print(sofa.state)
                break
        print(start_uid, max(users))
        start_uid += user_batch
        
    print("running time %.3e s" % (time.time() - begining_time))