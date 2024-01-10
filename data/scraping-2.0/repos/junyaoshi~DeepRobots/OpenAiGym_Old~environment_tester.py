import gym
# import Deep_Robot
from OpenAiGym.DiscreteDeepRobotEnv import DiscreteDeepRobotEnv
import math
from math import cos, sin, pi

# env = gym.make('Deep_Robot:DiscreteDeepRobot-v0')
env = DiscreteDeepRobotEnv(t_interval=1, a1=pi/8, a2=-pi/8)
env.reset()
for t in range(1000):
    action = (-0.5/10*sin(t/10+1), -0.5/10*sin(t/10))
    env.step(action)  # take a random action
    print('action taken(a1dot, a2dot): ', action)
    print('robot x y theta a1 a2: ', env.x, env.y, env.theta, env.a1, env.a2)
env.render()
env.close()
