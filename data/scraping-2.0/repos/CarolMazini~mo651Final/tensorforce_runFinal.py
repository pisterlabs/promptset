from gym import wrappers
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.execution import Runner
from tensorforce.agents import DQNAgent
from jaeger_p3dx import P3DX_Env

import numpy as np
import csv

def main():
    env = OpenAIGym("P3DX-v0")

    agent = DQNAgent(
        states=dict(type='float', shape=(80,80,4)),
        actions=dict(type='int', num_actions=7),
        network= [
                dict(
                    type="conv2d",
                    size=16,
                    window=[8,8],
                    stride=4,
                    activation="relu"
                ),
                dict(
                    type="conv2d",
                    size=32,
                    window=[4,4],
                    stride=2,
                    activation="relu"
                ),
                dict(
                    type="flatten"
                ),
                dict(
                    type="dense",
                    size=256
                )
        ],
        actions_exploration = dict(
            type="epsilon_decay",
            initial_epsilon=1.0,
            final_epsilon=0.1,
            timesteps=1000
        ),
        memory=dict(
                type="replay",
                capacity=1000,
                include_next_states=True
        ),
        update_mode = dict(
            unit="timesteps",
            batch_size=16,
            frequency=4
        ),
        discount = 0.99,
        entropy_regularization = None,
        double_q_model = True,
        optimizer = dict(
            type="adam",
            learning_rate=1e-4
        )
    )

    
    try:
        agent.restore_model(directory="data/", file="data-117246")
        print("Found data!")
    except:
        print("Can't load data")

    SAVE_INTERVAL = 10
    def episode_finished(r):
        #print(r.episode)
        if r.episode % SAVE_INTERVAL == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            print("Average of last {} rewards: {}\n".format(SAVE_INTERVAL, np.mean(r.episode_rewards[-SAVE_INTERVAL:])))

            r.agent.save_model(directory="data/data", append_timestep=True)

            with open("reward_history.csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                for reward in r.episode_rewards[-SAVE_INTERVAL:]:
                    writer.writerow([r.episode, reward])
        
            with open("episode_history.csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([r.episode, r.timestep])
        '''
        with open("individual_reward_history.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([r.episode, r.episode_rewards[-1]])
        '''
        return True

    runner = Runner(
        agent = agent,  # Agent object
        environment = env  # Environment object
    )

    max_episodes  = 10000
    max_timesteps = 50000000
    runner.run(max_timesteps,max_episodes, episode_finished=episode_finished)

    runner.close()

if __name__ == "__main__":
    main()
