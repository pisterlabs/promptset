# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
OpenAI gym execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import importlib
import json
import logging
import os
import time
import sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym_hvac
import gym

# python examples/openai_gym.py Pong-ram-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 50000 -m 2000

# python examples/openai_gym.py CartPole-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 2000 -m 200


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-i', '--import-modules', help="Import module(s) required for environment")
    parser.add_argument('-a', '--agent', help="Agent configuration file")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
    parser.add_argument('-f', '--full-day', action='store_true', default=False, help="Simulate a full Day")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False, help="Choose actions deterministically")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('--monitor', help="Save results to this directory")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('--visualize', action='store_true', default=False, help="Enable OpenAI Gym's visualization")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")
    parser.add_argument('-te', '--test', action='store_true', default=False, help="Test agent without learning.")
    parser.add_argument('-sl', '--sleep', type=float, default=None, help="Slow down simulation by sleeping for x seconds (fractions allowed).")
    parser.add_argument('--job', type=str, default=None, help="For distributed mode: The job type of this agent.")
    parser.add_argument('--task', type=int, default=0, help="For distributed mode: The task index of this agent.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # run the baseline version of the hvace for comparison
    baseIndoorTempArr, baseCostArr, baseRewardTempArr = baselineRun(args.max_episode_timesteps)

    if args.import_modules is not None:
        for module in args.import_modules.split(','):
            importlib.import_module(name=module)

    environment = OpenAIGym(
        gym_id=args.gym_id,
        monitor=args.monitor,
        monitor_safe=args.monitor_safe,
        monitor_video=args.monitor_video,
        visualize=args.visualize
    )

    if args.agent is not None:
        with open(args.agent, 'r') as fp:
            agent = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if args.network is not None:
        with open(args.network, 'r') as fp:
            network = json.load(fp=fp)
    else:
        network = None
        logger.info("No network configuration provided.")

    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network,
        )
    )

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(args.load)

    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    if args.debug:  # TODO: Timestep-based reporting
        report_episodes = 1
    else:
        report_episodes = 10

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))
    totalCostArr = []
    rewardArr = []
    temperatureArr = []
    episodeArr = []
    stepCountArr = []
    #fig, ax = plt.subplots()
    #plt.scatter(episodeArr, rewardArr, c='b', label='reward')
#    plt.scatter(episodeArr, totalCostArr, c='g', label='cost (cents)')
#    plt.scatter(episodeArr, temperatureArr, c='r', label='Temp(C)')
    #plt.scatter(episodeArr, stepCountArr, c='y', label='number of steps')
    #plt.legend(loc='lower left')    
    #ax.axhline((baseRewardTempArr[-1] - baseRewardTempArr[-2]), color='Yellow', lw=2, linestyle=':')
    
    plt.xlabel('number of episodes', fontsize=18)
    def episode_finished(r, id_):
        if r.episode % report_episodes == 0:
            episodeArr.append(r.episode)
            stepCountArr.append(r.episode_timestep)
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("*****************************************************************")
            logger.info("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
                r.agent.episode, r.episode_timestep, steps_per_second
            ))
            logger.info("*****************************************************************")
            electricalCost = r.environment.gym.hvacBuilding.CalculateElectricEneregyCost()
            gasCost = r.environment.gym.hvacBuilding.CalculateGasEneregyCost()
            rewardArr.append(r.episode_rewards[-1])
            temperatureArr.append(r.environment.gym.hvacBuilding.current_temperature)
            totalCostArr.append((electricalCost + gasCost)*100)

            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 5 rewards: {:0.2f}".format(sum(r.episode_rewards[-5:]) / min(5, len(r.episode_rewards))))
            logger.info("Current temperature: {}C".format(r.environment.gym.hvacBuilding.current_temperature))
            logger.info("Number of times Heating turned on: {}".format(r.environment.gym.hvacBuilding.building_hvac.NumberOfTimesHeatingTurnedOn))
            logger.info("Number of times Cooling turned on: {}".format(r.environment.gym.hvacBuilding.building_hvac.NumberOfTimesCoolingTurnedOn))
            logger.info("Total Time heating : {}".format(r.environment.gym.hvacBuilding.building_hvac.TotalDurationHeatingOn))
            logger.info("Total Time cooling : {}".format(r.environment.gym.hvacBuilding.building_hvac.TotalDurationCoolingOn))
            logger.info("*****************************************************************")
        if args.save and args.save_episodes is not None and not r.episode % args.save_episodes:
            logger.info("Saving agent to {}".format(args.save))
            r.agent.save_model(args.save)

        #plt.scatter(episodeArr, rewardArr, c='b', label='reward')
#        plt.scatter(episodeArr, totalCostArr, c='g', label='cost (cents)')
#        plt.scatter(episodeArr, temperatureArr, c='r', label='Temp(C)')
        #plt.scatter(episodeArr, stepCountArr, c='y', label='number of steps')
        #plt.pause(0.05)
        return True

    runner.run(
        num_timesteps=args.timesteps,
        num_episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished,
        testing=args.test,
        sleep=args.sleep
    )
	
    plt.style.use('seaborn')
    def mjrFormatter(x, pos):
        return str(datetime.timedelta(seconds=x))

    def addAxisLabels(plot: plt, yLabel:str, title:str, xlabel:str='Time of day', xAxisIsTime:bool = True):
        fig, ax = plot.subplots()
        if xAxisIsTime:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(mjrFormatter))
        plot.xlabel(xlabel, fontsize=18)
        plot.ylabel(yLabel, fontsize=18)
        plot.title(title, fontsize=20)
        return plot, ax

    # Print the graph for the Learning process
    learningPlot, axis = addAxisLabels(plt, 'Number of Steps per Episode', 'Reinforced Learning Progress', 'Number of Episodes', False)
    learningPlot.plot(episodeArr, stepCountArr, 'C3', label='Step Count')
    learningPlot.legend(loc='lower right')
    learningPlot.savefig('RL_Progress.svg')

    #plt.show()
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))
	
    indoorTempArr = []
    outdoorTempArr = []
    rewardSum = 0
    rewardTempArr = []
    costArr = []
    averageWattsPerSecArr = []
    timeOfDayInSecondsArr = []
    desiredTemperature = 20
    temperatureDelta = 2
    
	# setup the gym environment for one run
    numberOfSteps = args.max_episode_timesteps
    if args.full_day:
        numberOfSteps = 2880
    env = environment.gym
    # siulate 30 second intervals
    state = env.reset()
    for	i in range(numberOfSteps):
        timeOfDayInSecondsArr.append(i*env.env_step_interval)
        action = agent.act(state)
        state, reward, terminal, _ = env.step(action)
        rewardSum = rewardSum + reward
        rewardTempArr.append(rewardSum)
        indoorTempArr.append(state[1])
        outdoorTempArr.append(state[2])
        averageWattsPerSecArr.append(state[0])
        costArr.append(env.hvacBuilding.CalculateGasEneregyCost() + env.hvacBuilding.CalculateElectricEneregyCost())
        agent.observe(reward=reward, terminal=terminal)

    
    tempPlot, axis = addAxisLabels(plt, 'Temperature (C°)', '24 Hour HVAC House Temperature')
    axis.axhline(20, color='Yellow', lw=2, linestyle=':')
    tempPlot.plot(timeOfDayInSecondsArr, outdoorTempArr, 'C2', label='Outdoor Temp')
    tempPlot.plot(timeOfDayInSecondsArr, baseIndoorTempArr, 'b--', label='Standard Thermostat Indoor Temp')
    tempPlot.plot(timeOfDayInSecondsArr, indoorTempArr, 'C1', label='RL Indoor Temp')

    tempPlot.legend(loc='lower right')
    
	# save file
    tempPlot.savefig('indoorAndOutdoor_RL_Baseline_Comparison.svg')
	
    costPlot, axis = addAxisLabels(plt, 'Cost (US$)', '24 Hour HVAC Cost')
    costPlot.plot(timeOfDayInSecondsArr, costArr, 'C3', label='RL Total Cost')
    costPlot.plot(timeOfDayInSecondsArr, baseCostArr, 'b--', label='Standard Thermostat Total Cost')
    costPlot.legend(loc='lower right')
    costPlot.savefig('Total_Cost_RL_Baseline_Comparison.svg')
	
    rewardPlot, axis = addAxisLabels(plt, 'Reward (more is better)', '24 Hour HVAC Reward')
    rewardPlot.plot(timeOfDayInSecondsArr, rewardTempArr, 'C3', label='RL Reward')
    rewardPlot.plot(timeOfDayInSecondsArr, baseRewardTempArr, 'b--', label='Standard Thermostat Reward')
    rewardPlot.legend(loc='lower right')
    rewardPlot.savefig('Reward_RL_Baseline_Comparison.svg')
    runner.close()
    print(datetime.datetime.now())

def baselineRun(numberOfSteps):
    env = gym.make('Hvac-v0')
    done = False
    observation = env.reset()
    action = 0
    indoorTempArr = []
    outdoorTempArr = []
    rewardTempArr = []
    rewardSum = 0
    costArr = []
    averageWattsPerSecArr = []
    timeOfDayInSecondsArr = []
    desiredTemperature = 20
    temperatureDelta = 2
    
    # siulate 30 second intervals 
    for	i in range(numberOfSteps):
        timeOfDayInSecondsArr.append(i*env.env_step_interval)
        if not env.hvacBuilding.building_hvac.HeatingIsShuttingDown and env.hvacBuilding.building_hvac.HeatingIsOn and env.hvacBuilding.current_temperature > (desiredTemperature):
            #print("Turning the Heater Off")
            action = 0
    	
        if env.hvacBuilding.building_hvac.HeatingIsOn == False and env.hvacBuilding.current_temperature < (desiredTemperature - temperatureDelta):
            #print("Turning the Heater On")
            action = 1
    	
        if not env.hvacBuilding.building_hvac.HeatingIsOn and env.hvacBuilding.current_temperature > (desiredTemperature + temperatureDelta):
            #print("Turning the Cooling On")
            action = 2
    	
        if not env.hvacBuilding.building_hvac.HeatingIsOn and env.hvacBuilding.building_hvac.CoolingIsOn and env.hvacBuilding.current_temperature < desiredTemperature:
            #print("Turning the cooling off")
            action = 0
    	
        state, reward, done, info = env.step(action)
        rewardSum = rewardSum + reward
        rewardTempArr.append(rewardSum)
        indoorTempArr.append(state[1])
        outdoorTempArr.append(state[2])
        averageWattsPerSecArr.append(state[0])
        costArr.append(env.hvacBuilding.CalculateGasEneregyCost() + env.hvacBuilding.CalculateElectricEneregyCost())
    return indoorTempArr, costArr, rewardTempArr

if __name__ == '__main__':
    main()