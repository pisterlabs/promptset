#! /usr/bin/env python
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import rospy

rospy.init_node("fetch_moveit_test")
task_and_robot_environment_name = rospy.get_param(
    '/fetch/task_and_robot_environment_name')
# to register our task env to openai env.
# so that we don't care the output of this method for now.
env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

print("==== finished launching the simulation ====")

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        print("==== observation ====", observation)
        # action = env.action_space.sample()
        action = [0.7, 0.0, 0.85, 1.]  # sample action
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
