import os
import sys
import numpy as np
import time
import pommerman

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility

# Make sure you have tensorforce installed: pip install tensorforce
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.execution.threaded_runner import WorkerAgentGenerator

DEBUG = True
SHOULD_RENDER = True
NUM_EPISODES = 10
MAX_EPISODE_TIMTESTAMPS = 2000
MODEL_DIR = '/Users/voiceup/Git/cs221-pommerman/playground/notebooks/saved_ckpts/'
REPORT_EVERY_ITER = 100
SAVE_EVERY_ITER = 100

network_spec = [
          dict(type='dense', size=64),
          dict(type='dense', size=64)
      ]


def make_np_float(feature):
    return np.array(feature).astype(np.float32)


def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    return np.concatenate((
        board, bomb_blast_strength, bomb_life, position, ammo,
        blast_strength, can_kick, teammate, enemies))


class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass


def create_env_agent():
  # Instantiate the environment
  config = ffa_v0_fast_env()
  env = Pomme(**config["env_kwargs"])
  env.seed(0)

  # Create a Proximal Policy Optimization agent
  agent = PPOAgent(
      states=dict(type='float', shape=env.observation_space.shape),
      actions=dict(type='int', num_actions=env.action_space.n),
      network=network_spec,
      batching_capacity=1000,
      step_optimizer=dict(
          type='adam',
          learning_rate=1e-4
      ),

      # PGModel
      baseline_mode='network',
      baseline=dict(type='custom', network=[
          dict(type='dense', size=64),
          dict(type='dense', size=64)
      ]),
      baseline_optimizer=dict(
          type='adam',
          learning_rate=1e-4
      ),
  )

  # Add 3 random agents
  agents = []
  for agent_id in range(3):
      agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))
  print('asdfadfadf', config)
  # Add TensorforceAgent
  agent_id += 1
  agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))
  env.set_agents(agents)
  env.set_training_agent(agents[-1].agent_id)
  env.set_init_game_state(None)

  return (env, agent)


class WrappedEnv_MF(OpenAIGym):
    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize

    def execute(self, action):
        if self.visualize:
            self.gym.render()

        actions = self.unflatten_action(action=action)

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]

        if DEBUG:
          print('agent_state, terminal, agent_reward: ', agent_state, terminal, agent_reward)
          input('\n press any key to step forward \n')

        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = featurize(obs[3])
        return agent_obs


def episode_finished_runner(base_runner, task_id, interval=100):
    '''Callback for summary report.
    
       A function to be called once an episodes has finished. Should take
       a BaseRunner object and some worker ID (e.g. thread-ID or task-ID). Can decide for itself
       every how many episodes it should report something and what to report.
    '''
    if base_runner.global_episode % interval:
        return True
    
    end_episode = base_runner.global_episode
    start_episode = base_runner.global_episode
    print('=========================')
    print('Episode: {}'.format(base_runner.global_episode))
    print('Episode rewards', base_runner.episode_rewards[-interval:])
    print('Episode times', base_runner.episode_times[-interval:])
    print('=========================')
    return True


from tensorforce.execution import ThreadedRunner

class WrappedEnv(OpenAIGym):    
    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize
    
    def execute(self, action):
        print('Executing...')
        if self.visualize:
            self.gym.render()

        actions = self.unflatten_action(action=action)
            
        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward
    
    def reset(self):
        print('Resetting....')
        obs = self.gym.reset()
        agent_obs = featurize(obs[3])
        return agent_obs

def episode_finished(stats):
    print(
        "Thread {t}. Finished episode {ep} after {ts} timesteps. Reward {r}".
        format(t=stats['thread_id'], ep=stats['episode'], ts=stats['timestep'], r=stats['episode_reward'])
    )
    return True

def summary_report(r):
    et = time.time()
    print('=' * 40)
    print('Current Step/Episode: {}/{}'.format(r.global_step, r.global_episode))
    print('SPS: {}'.format(r.global_step / (et - r.start_time)))
    reward_list = r.episode_rewards
    if len(reward_list) > 0:
        print('Max Reward: {}'.format(np.max(reward_list)))
        print("Average of last 500 rewards: {}".format(sum(reward_list[-500:]) / 500))
        print("Average of last 100 rewards: {}".format(sum(reward_list[-100:]) / 100))
    print('=' * 40)
    
def main():
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)
    
    run_in_thread = True
    
    # Create a set of agents (exactly four)
    env, agent = create_env_agent()
    wrapped_env = WrappedEnv(env, True)

    agents=[agent]
    for i in range(2):
        worker = WorkerAgentGenerator(type(agent))(
            states=agent.states,
            actions=agent.actions,
            network=network_spec,
            model=agent.model,
        )
        agents.append(worker)
        
    if run_in_thread:
        threaded_runner = ThreadedRunner(
            agents,
            [wrapped_env]*3,
#             repeat_actions=1,
#             save_path='save_path/',
#             save_episodes=2
        )
        print("Starting {agent} for Environment '{env}'".format(agent=agent, env=wrapped_env))
#         threaded_runner.run(episodes=1, summary_interval=1, episode_finished=episode_finished, summary_report=summary_report)
        threaded_runner.run(episodes=2, episode_finished=episode_finished, max_episode_timesteps=MAX_EPISODE_TIMTESTAMPS)

        try:
            threaded_runner.close()
        except AttributeError as e:
            print('AttributeError: ', e)
    else:
        runner = Runner(agent=agent, environment=wrapped_env)
        runner.run(
          episodes=10,
          max_episode_timesteps=MAX_EPISODE_TIMTESTAMPS,
          episode_finished=episode_finished_runner
        )

        try:
            runner.close()
        except AttributeError as e:
            print('AttributeError: ', e)

if __name__ == '__main__':
    main()
