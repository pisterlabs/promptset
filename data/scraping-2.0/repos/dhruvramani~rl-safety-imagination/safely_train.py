# Inspired from OpenAI Baselines. This uses the same design of having an easily
# substitutable generic policy that can be trained. This allows to easily
# substitute in the I2A policy as opposed to the basic CNN one.

import copy
import argparse
import statistics 
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from safe_grid_gym.envs.gridworlds_env import GridworldEnv

#from i2a import I2aPolicy
from utils import SubprocVecEnv
from discretize_env import CONTROLS
from a2c import CnnPolicy, get_actor_critic
from trajectory import *

ENV_NAME = "side_effects_sokoban"
N_ENVS = 1
N_STEPS = 9
END_REWARD = 49
S_ALPHAS =  [0.1, 0.3, 1.0, 2.0, 3.0, 10.0, 30.0, 100.0, 300.0]
s_alpha = 2.0 #S_ALPHAS[4]
DEBUG = False

# For early stopping
EARLY_STOPPING = False
REW_HIST = 3 
EARLY_STOP_THRESH = 1.5

# Total number of iterations (taking into account number of environments and
# number of steps). You wish to train for.
TOTAL_TIMESTEPS = int(500)

GAMMA = 0.99

LOG_INTERVAL = 100
SAVE_INTERVAL = 100

# Where you want to save the weights
SAVE_PATH = 'safe_a2c_weights/{:.1f}'.format(s_alpha)

def discount_with_dones(rewards, dones, GAMMA):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + GAMMA * r * (1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_env():
    def _thunk():
        env = GridworldEnv(ENV_NAME)
        return env

    return _thunk

def a2c_safe_action(tree, action, base_state, actor_critic):
    is_end = False
    try :
        next_node = tree.children[action[0]]
        is_end = next_node.imagined_reward == END_REWARD
    except AttributeError:
        next_node = None
    if(is_end == False and search_node(next_node, base_state) == False):
        try:
            action = safe_action(actor_critic, tree, base_state, action[0])
        except:
            pass
    return action


def train(policy, save_name, s_alpha, load_count = 0, summarize=True, load_path=None, log_path = './logs', safety=True):
    envs = make_env()() #for i in range(N_ENVS)]
    #envs = SubprocVecEnv(envs)
    with open("./unsafe_state_count_{}.txt".format(safety), "w+") as f:
        pass

    ob_space = envs.observation_space.shape
    nc, nw, nh = ob_space
    ac_space = envs.action_space

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    actor_critic = get_actor_critic(sess, N_ENVS, N_STEPS, ob_space,
            ac_space, policy, summarize)
    if load_path is not None:
        actor_critic.load(load_path)
        print('Loaded a2c')

    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_path, graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    batch_ob_shape = (N_ENVS * N_STEPS, nw, nh, nc)
    dones = False #for _ in range(N_ENVS)]
    nbatch =  N_STEPS # * N_Envs
    episode_rewards = np.zeros((1, ))
    final_rewards   = np.zeros((1, ))
    last_rews = [0] * REW_HIST

    # Safety part
    obs = envs.reset()
    ob_np = obs.reshape(nc, nw, nh)
    
    base_state = copy.deepcopy(ob_np).reshape(nc, nw, nh)
    base_state[np.where(base_state == 2.0)] = 1.0
    print(base_state)
    base_tree = generate_tree(sess, ob_np)

    for update in tqdm(range(load_count + 1, TOTAL_TIMESTEPS + 1)):
        # mb stands for mini batch
        unsafe_state_count = 0
        tree = copy.deepcopy(base_tree)
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        
        for n in range(N_STEPS):
            ob_np = obs.reshape(nc, nw, nh)

            unsafe_state = ob_np.reshape(nw, nh) 
            x, y = np.where(unsafe_state == 4.0)
            if(x == 3 and y == 2):
                unsafe_state_count += 1

            if(update % LOG_INTERVAL == 0 and DEBUG == True):
                print("-- State ---")
                print(ob_np)
                print("-- Imagined State --")
                print(tree.imagined_state.reshape(nc, nw, nh))

            ac_ob = ob_np.reshape(1, nw, nh, nc)
            actions, values, _ = actor_critic.act(ac_ob)
            if(safety):
                actions = a2c_safe_action(tree, actions, base_state, actor_critic)

            mb_obs.append(ob_np)
            mb_actions.append(actions[0])
            mb_values.append(values)
            mb_dones.append(dones)

            if(update % LOG_INTERVAL == 0 and DEBUG == True):
                print("Action : ", CONTROLS[actions[0]], " - Safe :", bool(safe[0])," - Done : ", dones[0])
                _ = input("")

            obs, rewards, dones, _ = envs.step(actions[0])
            ob_np = ob_np.reshape(nc, nw, nh)
            
            tree = get_node(base_tree, ob_np) 

            #rewards = [rewards[i] - s_alpha * (1 - safe[i]) for i in range(len(rewards))]
            episode_rewards += rewards
            masks = 1 - int(dones)
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            mb_rewards.append(rewards)

        with open("./unsafe_state_count_{}.txt".format(safety), "a+") as f:
            f.write("{}\n".format(unsafe_state_count))
            unsafe_state_count = 0

        mb_dones.append(dones)
        obs = envs.reset()
        tree = copy.deepcopy(base_tree)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).reshape(batch_ob_shape) #.swapaxes(1, 0).reshape(batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)#.swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)#.swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32)#.swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)#.swapaxes(1, 0)
        mb_masks = mb_dones[:-1]
        mb_dones = mb_dones[1:]

        ac_ob = ob_np.reshape(1, nw, nh, nc)
        last_values = actor_critic.critique(ac_ob).tolist()

        #discount/bootstrap off value fn
        #for n, (rewards, value) in enumerate(zip(mb_rewards, last_values)):
        rewards = mb_rewards.tolist()
        d = mb_dones.tolist()
        value = last_values
        if d[-1] == 0:
            rewards = discount_with_dones(rewards+value, d+[0], GAMMA)[:-1]
        else:
            rewards = discount_with_dones(rewards, d, GAMMA)
        mb_rewards = np.array(rewards)

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()

        if summarize:
            loss, policy_loss, value_loss, policy_entropy, _, summary = actor_critic.train(mb_obs,
                    mb_rewards, mb_masks, mb_actions, mb_values, update,
                    summary_op)
            writer.add_summary(summary, update)
        else:
            loss, policy_loss, value_loss, policy_entropy, _ = actor_critic.train(mb_obs,
                    mb_rewards, mb_masks, mb_actions, mb_values, update)

        if update % LOG_INTERVAL == 0 or update == 1:
            print('%i - %.1f => Policy Loss : %.4f, Value Loss : %.4f, Policy Entropy : %.4f, Final Reward : %.4f' % (update, s_alpha, policy_loss, value_loss, policy_entropy, final_rewards.mean()))
            if(EARLY_STOPPING and update != 1 and abs(final_rewards.mean() - statistics.mean(last_rews)) < EARLY_STOP_THRESH):
                print('Training done - Saving model')
                actor_critic.save(SAVE_PATH, save_name + '_' + str(update) + '.ckpt')
                with open("./logs_alpha.txt", "a+") as f:
                    f.write("{:.1f} - {:.4f}\n".format(s_alpha, max(last_rews)))
                break
            _ = last_rews.pop(0)
            last_rews.append(final_rewards.mean())

        if update % SAVE_INTERVAL == 0:
            print('Saving model')
            actor_critic.save(SAVE_PATH, save_name + '_' + str(update) + '.ckpt')

        actor_critic.save(SAVE_PATH, save_name + '_done.ckpt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', help='Algorithm to train a2c (or something else in the future)')
    args = parser.parse_args()

    if args.algo == 'a2c':
        policy = CnnPolicy
    else:
        raise ValueError('Must specify the algo name as either a2c or (something else in the future)')

    #for s_alpha in S_ALPHAS:
        #tf.reset_default_graph()
    train(policy, args.algo + "{:.1f}".format(s_alpha), s_alpha=s_alpha, summarize=True, log_path="safe_" + args.algo + '_logs/'+ "{:.1f}".format(s_alpha))
