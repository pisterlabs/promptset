 # -*- coding: utf-8 -*-
'''

***Author:*** Federico Berto

============= Thesis Project for University of Bologna =============
Reinforcement Learning: a Preliminary Study on Vision-Based Control

A special thanks goes to gi`Adam Paszke <https://github.com/apaszke>`_, 
for a first implementation of the DQN algorithm with vision input in
the Cartpole-V0 environment from OpenAI Gym.
`Gym website <https://gym.openai.com/envs/CartPole-v0>`__.

The goal of this project is to design a control system for stabilizing a
Cart and Pole using Deep Reinforcement Learning, having only images as 
control inputs. We implement the vision-based control using the DQN algorithm
combined with Convolutional Neural Network for Q-values approximation.

The last two frames of the Cartpole are used as input, cropped and processed 
before using them in the Neural Network. In order to stabilize the training,
we use an experience replay buffer as shown in the paper "Playing Atari with
Deep Reinforcement Learning:
 <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>__.

Besides, a target network to further stabilize the training process is used.
make the training not converge, we set a threshold for stopping training
when we detect stable improvements: this way we learn optimal behavior
without saturation. 

The GUI is a handi tool for saving and loading trained models, and also for
training start/stop. Models and Graphs are saved in Vision_Carpole/save_model
and Vision_Cartpole/save_graph respectively.

'''

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque
import tkinter

############ HYPERPARAMETERS ##############

BATCH_SIZE = 128 # original = 128
GAMMA = 0.999 # original = 0.999
EPS_START = 0.9 # original = 0.9
EPS_END = 0.05 # original = 0.05
EPS_DECAY = 5000 # original = 200
TARGET_UPDATE = 50 # original = 10
MEMORY_SIZE = 100000 # original = 10000
END_SCORE = 200 # 200 for Cartpole-v0
TRAINING_STOP = 142 # threshold for training stop
N_EPISODES = 50000 # total episodes to be run
LAST_EPISODES_NUM = 20 # number of episodes for stopping training
FRAMES = 2 # state is the number of last frames: the more frames, 
# the more the state is detailed (still Markovian)
RESIZE_PIXELS = 60 # Downsample image to this number of pixels

# ---- CONVOLUTIONAL NEURAL NETWORK ----
HIDDEN_LAYER_1 = 16
HIDDEN_LAYER_2 = 32 
HIDDEN_LAYER_3 = 32
KERNEL_SIZE = 5 # original = 5
STRIDE = 2 # original = 2
# --------------------------------------

GRAYSCALE = True # False is RGB
LOAD_MODEL = False # If we want to load the model, Default= False
USE_CUDA = False # If we want to use GPU (powerful one needed!)
############################################

graph_name = 'Cartpole_Vision_Stop-' + str(TRAINING_STOP) + '_LastEpNum-' + str(LAST_EPISODES_NUM)


# GUI for saving models with Tkinter
FONT =  "Fixedsys 12 bold" #GUI font
save_command1, save_command2, save_command3 = 0, 0, 0
load_command_1, load_command_2, load_command_3 = 0, 0, 0
resume_command, stop_command = 0,0
window = tkinter.Tk()
window.lift()
window.attributes("-topmost", True)
window.title("DQN-Vision Manager")
lbl = tkinter.Label(window, text="Manage training -->")
lbl.grid(column=0, row=0)

def clicked1():
    global save_command1
    lbl.configure(text="Model saved in slot 1!")
    save_command1 = True
btn1 = tkinter.Button(window, text="Save 1", font= FONT, command=clicked1,  bg= "gray")
btn1.grid(column=1, row=0)

def clicked2():
    global save_command2
    lbl.configure(text="Model saved in slot 2!")
    save_command2 = True
btn2 = tkinter.Button(window, text="Save 2", font= FONT,command=clicked2, bg= "gray")
btn2.grid(column=2, row=0)

def clicked3():
    global save_command3
    lbl.configure(text="Model saved in slot 3!")
    save_command3 = True
btn3 = tkinter.Button(window, text="Save Best", font= FONT, command=clicked3,  bg= "gray")
btn3.grid(column=3, row=0)

def clicked_load1():
    global load_command_1
    lbl.configure(text="Model loaded from slot 1!")
    load_command_1 = True
load_btn1 = tkinter.Button(window, text="Load 1", font= FONT, command=clicked_load1,  bg= "blue")
load_btn1.grid(column=1, row=1)

def clicked_load2():
    global load_command_2
    lbl.configure(text="Model loaded from slot 2!")
    load_command_2 = True
load_btn2 = tkinter.Button(window, text="Load 2", font= FONT, command=clicked_load2,  bg= "blue")
load_btn2.grid(column=2, row=1)

def clicked_load3():
    global load_command_3
    lbl.configure(text="Model loaded from slot 3!")
    load_command_3 = True
load_btn3 = tkinter.Button(window, text="Load Best", font= FONT, command=clicked_load3,  bg= "blue")
load_btn3.grid(column=3, row=1)

def clicked_resume():
    global resume_command
    lbl.configure(text="Training resumed!")
    resume_command = True
resume_btn = tkinter.Button(window, text="Resume Training", font= FONT, command=clicked_resume,  bg= "green")
resume_btn.grid(column=1, row=2)

def clicked_stop():
    global stop_command
    lbl.configure(text="Training stopped!")
    stop_command = True
stop_btn = tkinter.Button(window, text="Stop Training", font= FONT, command=clicked_stop,  bg= "red")
stop_btn.grid(column=3, row=2)


# Settings for GRAYSCALE / RGB
if GRAYSCALE == 0:
    resize = T.Compose([T.ToPILImage(), 
                    T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                    T.ToTensor()])
    
    nn_inputs = 3*FRAMES  # number of channels for the nn
else:
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                    T.Grayscale(),
                    T.ToTensor()])
    nn_inputs =  FRAMES # number of channels for the nn

                    
stop_training = False 

env = gym.make('CartPole-v0').unwrapped 

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# If gpu is to be used
device = torch.device("cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Memory for Experience Replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None) # if we haven't reached full capacity, we append a new transition
        self.memory[self.position] = Transition(*args)  
        self.position = (self.position + 1) % self.capacity # e.g if the capacity is 100, and our position is now 101, we don't append to
        # position 101 (impossible), but to position 1 (its remainder), overwriting old data

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) 

    def __len__(self): 
        return len(self.memory)

# Build CNN
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(nn_inputs, HIDDEN_LAYER_1, kernel_size=KERNEL_SIZE, stride=STRIDE) 
        self.bn1 = nn.BatchNorm2d(HIDDEN_LAYER_1)
        self.conv2 = nn.Conv2d(HIDDEN_LAYER_1, HIDDEN_LAYER_2, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn2 = nn.BatchNorm2d(HIDDEN_LAYER_2)
        self.conv3 = nn.Conv2d(HIDDEN_LAYER_2, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn3 = nn.BatchNorm2d(HIDDEN_LAYER_3)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        nn.Dropout()
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

# Cart location for centering image crop
def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

# Cropping, downsampling (and Grayscaling) image
def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
if GRAYSCALE == 0:
    plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
            interpolation='none')
else:
    plt.imshow(get_screen().cpu().squeeze(0).permute(
        1, 2, 0).numpy().squeeze(), cmap='gray')
plt.title('Example extracted screen')
plt.show()




eps_threshold = 0.9 # original = 0.9

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
print("Screen height: ", screen_height," | Width: ", screen_width)

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if LOAD_MODEL == True:
    policy_net_checkpoint = torch.load('save_model/policy_net_best3.pt') # best 3 is the default best
    target_net_checkpoint = torch.load('save_model/target_net_best3.pt')
    policy_net.load_state_dict(policy_net_checkpoint)
    target_net.load_state_dict(target_net_checkpoint)
    policy_net.eval()
    target_net.eval()
    stop_training = True # if we want to load, then we don't train the network anymore

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0

# Action selection , if stop training == True, only exploitation
def select_action(state, stop_training):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # print('Epsilon = ', eps_threshold, end='\n')
    if sample > eps_threshold or stop_training:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


# Plotting
def plot_durations(score):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    episode_number = len(durations_t) 
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), label= 'Score')
    matplotlib.pyplot.hlines(195, 0, episode_number, colors='red', linestyles=':', label='Win Threshold')
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        last100_mean = means[episode_number -100].item()
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label= 'Last 100 mean')
        print('Episode: ', episode_number, ' | Score: ', score, '| Last 100 mean = ', last100_mean)
    plt.legend(loc='upper left')
    #plt.savefig('./save_graph/cartpole_dqn_vision_test.png') # for saving graph with latest 100 mean
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig('save_graph/' + graph_name)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

# Training 
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # torch.cat concatenates tensor sequence
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    plt.figure(2)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

mean_last = deque([0] * LAST_EPISODES_NUM, LAST_EPISODES_NUM)

# Button click checking

def check_button():
    global save_command1
    global save_command2
    global save_command3
    global load_command_1
    global load_command_2
    global load_command_3
    global resume_command
    global stop_command
    global stop_training

# Saving the weights
    if save_command1 == True:
        torch.save(policy_net.state_dict(), 'save_model/policy_net_best1.pt') # we save the trained policy model
        torch.save(target_net.state_dict(), 'save_model/target_net_best1.pt') # we save the trained target model
        save_command1 = False
    if save_command2 == True:
        torch.save(policy_net.state_dict(), 'save_model/policy_net_best2.pt') # we save the trained policy model
        torch.save(target_net.state_dict(), 'save_model/target_net_best2.pt') # we save the trained target model
        save_command2 = False
    if save_command3 == True:
        torch.save(policy_net.state_dict(), 'save_model/policy_net_best3.pt') # we save the trained policy model
        torch.save(target_net.state_dict(), 'save_model/target_net_best3.pt') # we save the trained target model
        save_command3 = False
# Loading the Weights  
    if load_command_1 or load_command_2 or load_command_3:
        if load_command_1:   
            policy_net_checkpoint = torch.load('save_model/policy_net_best1.pt')
            target_net_checkpoint = torch.load('save_model/target_net_best1.pt')
            load_command_1 = False
        if load_command_2:
            policy_net_checkpoint = torch.load('save_model/policy_net_best2.pt')
            target_net_checkpoint = torch.load('save_model/target_net_best2.pt')
            load_command_2 = False
        if load_command_3:   
            policy_net_checkpoint = torch.load('save_model/policy_net_best3.pt')
            target_net_checkpoint = torch.load('save_model/target_net_best3.pt')
            load_command_3 = False

        policy_net.load_state_dict(policy_net_checkpoint)
        target_net.load_state_dict(target_net_checkpoint)
        policy_net.eval()
        target_net.eval()
        stop_training = True # if we want to load, then we don't train the network anymore

# Training Start/Stop
    if resume_command == True:
        stop_training = False
        resume_command = False
    if stop_command == True:
        stop_training = True
        stop_command = False

window.attributes("-topmost", False)


# MAIN LOOP

for i_episode in range(N_EPISODES):
    # Initialize the environment and state
    env.reset()
    init_screen = get_screen()
    screens = deque([init_screen] * FRAMES, FRAMES)
    state = torch.cat(list(screens), dim=1)

    for t in count():

        # Select and perform an action
        action = select_action(state, stop_training)
        state_variables, _, done, _ = env.step(action.item())

        # Observe new state
        screens.append(get_screen())
        next_state = torch.cat(list(screens), dim=1) if not done else None

        # Reward modification for better stability
        x, x_dot, theta, theta_dot = state_variables
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        reward = torch.tensor([reward], device=device)
        if t >= END_SCORE-1:
            reward = reward + 20
            done = 1
        else: 
            if done:
                reward = reward - 20 

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        
        # Perform one step of the optimization (on the target network)
        if done:
            check_button() # We check the GUI for inputs
            episode_durations.append(t + 1)
            plot_durations(t + 1)
            mean_last.append(t + 1)
            mean = 0
            for i in range(LAST_EPISODES_NUM):
                mean = mean_last[i] + mean
            mean = mean/LAST_EPISODES_NUM
            if mean < TRAINING_STOP and stop_training == False:
                optimize_model()
            else:
                stop_training = 1
            break
            
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()