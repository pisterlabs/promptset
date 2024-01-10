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
        agent.restore_model(directory="modelo/", file="data-129235")
        print("Found data!")
    except Exception as e:
        print(e)
        print("Can't load data")


    print("Starting execution")
    state = env.reset()
    agent.reset()
    try:
        while True:
            # Get action - no exploration and no observing
            action = agent.act(state, deterministic=True, independent=True)
            print(action)

            # Execute action in the environment
            state, terminal_state, reward = env.execute(action)

            if terminal_state:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("Terminal state", terminal_state)
        state = env.reset()
        agent.reset()

if __name__ == "__main__":
    main()
