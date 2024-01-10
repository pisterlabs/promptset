#!/usr/bin/env python

'''
    Training code made by Ricardo Tellez <rtellez@theconstructsim.com>
    Based on many other examples around Internet
    Visit our website at www.theconstruct.ai
'''
import os
import gym
import time
import numpy
import random
import time
import qlearn
import logging
import json
from tensorforce.agents import Agent
from gym import wrappers
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.execution import Runner

# ROS packages required
import rospy
import rospkg

# import our training environment
import myquadcopter_env


if __name__ == '__main__':
    

    rospy.init_node('drone_gym', anonymous=True)


    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('drone_training')
    outdir = pkg_path + '/training_results'
    
    rospy.loginfo ( "Monitor Wrapper started")

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    environment = OpenAIGym(
        gym_id='QuadcopterLiveShow-v0',
        monitor='output',
        monitor_safe=False,
        monitor_video=False,
        visualize=True
    )
    print os.getcwd()
    with open('/home/user/catkin_ws/src/drone_training/drone_training/configs/trpo.json', 'r') as fp:
        agent = json.load(fp=fp)

    with open('/home/user/catkin_ws/src/drone_training/drone_training/configs/mynet.json', 'r') as fp:
        network = json.load(fp=fp)

    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network,
        )
    )
    if rospy.get_param("/load"):
        load_dir = os.path.dirname(rospy.get_param("/load"))
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(rospy.get_param("/load"))

    if rospy.get_param("/save"):
        save_dir = os.path.dirname(rospy.get_param("/save"))
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))

    if rospy.get_param("/debug"):
        print("-" * 16)
        print("Configuration:")
        print(agent)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    if rospy.get_param("/debug"):  # TODO: Timestep-based reporting
        report_episodes = 1
    else:
        report_episodes = 100

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    def episode_finished(r, id_):
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            print("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
                r.agent.episode, r.episode_timestep, steps_per_second
            ))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            print("Average of last 500 rewards: {:0.2f}".
                        format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            print("Average of last 100 rewards: {:0.2f}".
                        format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
            print("{},{}".format(r.agent.episode,r.episode_rewards[-1]))
            with open("data.csv", "a") as myfile:
                myfile.write("{},{}".format(r.agent.episode,r.episode_rewards[-1]) )
        if rospy.get_param("/save") and rospy.get_param("/save_episodes") is not None and not r.episode % rospy.get_param("/save_episodes"):
            print("Saving agent to {}".format(rospy.get_param("/save")))
            r.agent.save_model(rospy.get_param("/save"))

        return True

    runner.run(
        max_episode_timesteps=rospy.get_param("/nsteps"),
        num_episodes=rospy.get_param("/nepisodes"),
        deterministic=False,
        episode_finished=episode_finished
    )
    runner.close()

    print("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))
