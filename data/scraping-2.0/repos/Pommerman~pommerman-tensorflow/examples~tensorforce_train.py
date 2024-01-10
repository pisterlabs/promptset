# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:01:53 2019

@author: Myron Kwan


python tensorforce_train.py --agents "tensorforce::ppo,random::,random::,random::" --config=PommeFFACompetition-v0 --episodes 100 --batch-size 1 --modelname testing 

"""

import atexit
import functools
import os 
import argparse

import tensorforce
from tensorforce.execution import ParallelRunner
from tensorforce.environments.openai_gym import OpenAIGym
import gym

from pommerman import helpers,make
from pommerman.agents import TensorForceAgent

import timeit
import matplotlib.pyplot as plt
import pickle
import numpy as np

def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]


class WrappedEnv(OpenAIGym):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''

    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize
        self.thread=None
    def execute(self, actions):
        if self.visualize:
            self.gym.render()

        actions = self.unflatten_action(action=actions)

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[3])
        return agent_obs

def main():
    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--game", default="pommerman", help="Game to choose.")
    parser.add_argument(
        "--config",
        default='PommeFFACompetition-v0',
        help="Configuration to execute. See env_ids in "
        "configs.py for options. default is 1v1")
    parser.add_argument(
        "--agents",
        default="tensorforce::ppo,test::agents.SimpleAgent,"
        "test::agents.SimpleAgent,test::agents.SimpleAgent",
        #default="tensorforce::ppo,test::agents.RandomAgent,"
        #"test::agents.RandomAgent,test::agents.RandomAgent",
        help="Comma delineated list of agent types and docker "
        "locations to run the agents.")
        #agent in position 1

    parser.add_argument(
        "--agent_env_vars",
        help="Comma delineated list of agent environment vars "
        "to pass to Docker. This is only for the Docker Agent."
        " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
        "would send two arguments to Docker Agent 0 and one to"
        " Docker Agent 3.",
        default="")
    parser.add_argument(
        "--record_pngs_dir",
        default=None,
        help="Directory to record the PNGs of the game. "
        "Doesn't record if None.")
    parser.add_argument(
        "--record_json_dir",
        default=None,
        help="Directory to record the JSON representations of "
        "the game. Doesn't record if None.")
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--game_state_file",
        default=None,
        help="File from which to load game state. Defaults to "
        "None.")
    parser.add_argument(
            '--batch-size', # This doesn't change batch-size in tensorforce_agent.py
            default=10,
            type=int,
            help='average reward visualization by batch size. default=100 episodes'
            )
    parser.add_argument(
            '--episodes',
            default=10,
            type=int,
            help='number of training episodes, default=1000. must be divisible by batch_size'
            )
    parser.add_argument(
            '--modelname',
            default='default',
            help='name of model file savename, timesteps wil be appended. default= default'
            )
    parser.add_argument(
            '--loadfile',
            default=None,
            help='name of model you want to load'
            )
    parser.add_argument(
            '--numprocs',
            default=12,
            type=int,
            help='num parallel processes. default=12'
            )
    args = parser.parse_args()

    config = args.config
    record_pngs_dir = args.record_pngs_dir
    record_json_dir = args.record_json_dir
    agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file
    num_procs=args.numprocs

    #variables    
    save_path='saved_models/'
    model_name=args.modelname
    batch_size=args.batch_size
    num_episodes=args.episodes
    assert(num_episodes%batch_size==0)
    
    agents = [
        helpers.make_agent_from_string(agent_string, agent_id)
        for agent_id, agent_string in enumerate(args.agents.split(","))
    ]
    env = make(config, agents, game_state_file)
    training_agent = None

    for agent in agents:
        if type(agent) == TensorForceAgent:
            training_agent = agent
            env.set_training_agent(agent.agent_id)
            break
        
    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)   

    agent = training_agent.initialize(env,num_procs, 
                # summarizer={'directory': 'tensorforce_agent', 'labels': 'graph, losses'},
                #saver={'directory': './'+save_path, 'filename': model_name,'append_timesteps': True}
                )

    # USHA Model should load automatically as saver is provided.
    if args.loadfile:
        agent.restore(directory=save_path,filename=args.loadfile)

    atexit.register(functools.partial(clean_up_agents, agents))
     
    wrapped_envs=[]
    for i in range(num_procs):
        wrapped_envs.append(WrappedEnv(env, visualize=args.render))
         
 
    # wrapped_env=WrappedEnv(env,visualize=args.render)

    runner_time = timeit.default_timer()

    #load history.pickle

    if args.loadfile:
         try :
             handle = open(save_path+args.modelname+'-history.pkl','rb')
             history=pickle.load(handle)
         except:
             history=None
    else:
         history=None


    runner = ParallelRunner(agent=agent, environments=wrapped_envs)
    # runner = Runner(agent=agent, environment=wrapped_env)

    
    num_episodes+=runner.global_episodes #runner trains off number of global episodes
    '''
    if you trained 100 episodes, num_episodes needs to be 200 if you want to train another 100
    '''
    
    runner.run(num_episodes=num_episodes, max_episode_timesteps=2000)
    
    print(runner.episode_rewards)
    
    if history:
        history['episode_rewards'].extend(runner.episode_rewards)
        history['episode_timesteps'].extend(runner.episode_timesteps)
        history['episode_seconds'].extend(runner.episode_seconds)
        history['episode_agent_seconds'].extend(runner.episode_agent_seconds)
    else:
        history={}
        history['episode_rewards']=runner.episode_rewards
        history['episode_timesteps']=runner.episode_timesteps
        history['episode_seconds']=runner.episode_seconds
        history['episode_agent_seconds']=runner.episode_agent_seconds
        
    with open(save_path+model_name+'-history.pkl','wb') as handle:
        pickle.dump(history,handle)
    # USHA Model should save automatically as saver is provided.
    agent.save(directory=save_path,filename=model_name+str(runner.global_episodes),append_timestep=False)
    print('Runner time: ', timeit.default_timer() - runner_time)

    plt.plot(np.arange(0,int(len(history['episode_rewards'])/batch_size)),np.mean(np.asarray(history['episode_rewards']).reshape(-1,batch_size),axis=1))
    plt.title('average rewards per batch of episodes')
    plt.ylabel('average reward')
    plt.xlabel('batch of ' +str(batch_size)+' episodes')
    plt.show()
    

    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == "__main__":
    main()
