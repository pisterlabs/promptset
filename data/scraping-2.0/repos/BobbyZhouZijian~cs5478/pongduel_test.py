import gym
import time
import openai
import numpy as np
from ma_gym.wrappers import Monitor

# set the openai api key
openai.api_key = "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" # your openai api key

env = gym.make('ma_gym:PongDuel-v0')
env = Monitor(env, directory='recordings',video_callable=lambda episode_id: True)
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

def get_prompt(states, player):
    state0 = states[player]
    direction = np.argmax(state0[4:])

    if player == 0:
        if direction == 0:
            direction = "left-top"
        elif direction == 1:
            direction = "left"
        elif direction == 2:
            direction = "left-down"
        elif direction == 3:
            direction = "right-down"
        elif direction == 4:
            direction = "right"
        elif direction == 5:
            direction = "right-top"
    else:
        if direction == 0:
            direction = "right-top"
        elif direction == 1:
            direction = "right"
        elif direction == 2:
            direction = "right-down"
        elif direction == 3:
            direction = "left-down"
        elif direction == 4:
            direction = "left"
        elif direction == 5:
            direction = "left-top"

    prompt = """
    Given at a certain round, the ball is at ({:.1f}, {:.1f}) and moving {}, your paddle is at at ({:.1f}, {:.1f}). How should you move your paddle?
    Answer "up" or "down" or "noop" with justication.
    """.format(state0[3], 1 - state0[2], direction, state0[1], 1 - state0[0])
    return prompt

def get_exlpainer_prompt(text):
    prompt = """
{}

The above is the description of a player's decision about the action in a Pong game. What decision did the player make? 
Answer only one word of the following: noop, up, down without any other justification.
""".format(text)
    return prompt


obs_n = env.reset()
iter = 50
records = []
round_idx = -1
for _ in range(iter):
    env.render()
    state = env.get_agent_obs()
    # create openai chat agent
    prompt = get_prompt(state, 0)
    context = '''
    You are an expert in planning and decision making. Now you need to play a two player Pong game.

    You and another player will paddle a ball, and the goal is to make the opponent miss the ball.
    There is a 2d board where the direction is represent as 2d vector, and the location is also in 2d coordinate.

    The movement of the ball follows the following rules:
    1. When the ball is moving left or right, the position will be changed by (-0.1, 0) or (0.1, 0)
    2. When the ball is moving left-top or left-bottom, the position will be changed by (-0.1, 0.1) or (-0.1, 0.1)
    3. When the ball is moving right-top or right-bottom, the position will be changed by (0.1, 0.1) or (0.1, -0.1)
    4. When the ball reaches the boundary x=0 or x=1, if the original direction is (x, y), the new direction will be (-x, y)
    5. When the ball reaches the boundary y=0 or y=1, if the original direction is (x, y), the new direction will be (x, -y)

    The movement of the players follows the following rules:
    1. Your position is represented as (x, y), and x = 0 all the time; the adversial player's position is represented as (x, y), and x = 1 all the time
    2. When the player is moving up, the position will be changed by (0, 0.1)
    3. When the player is moving down, the position will be changed by (0, -0.1)
    4. When the player noop, the position will not be changed
    5. The paddle is represented as a range between (x, y-0.05) to (x, y+0.05)

    The rule of win and lose:
    1. Suppose your paddile is at(0, y-0.05) to (0, y+0.05), the ball's location is (a, b), and the ball's direction is (1, y'-0.05) to (1, y'+0.05)
    2. When a = 0, if b is out of y-0.05 and y+0.05, you will lose the game
    3. When a = 1, if b is out of y'-0.05 and y'+0.05, you will win the game

    Your goal is to win the game.
    '''
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": prompt},
        ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.8,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # get the text from the response
    action = response.choices[0].message['content']
    print(action)
    action = get_exlpainer_prompt(action)
    messages = [
        {"role": "system", "content": "You are an expert in summarizing things."},
        {"role": "user", "content": action},
        ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )
    action = response.choices[0].message['content']
    print(action)
    if "noop" in action.lower():
        action_1 = 0
    elif "up" in action.lower():
        action_1 = 1
    else:
        action_1 = 2

    print(f'---------End of Iter {_}----------')
    # filter the keyword "up" or "down" from the response
    # check if the word "up" is in the response
    
    action_2 = np.random.randint(0, 3)
    # take the actions
    action_n = [action_1, action_2]
    for _ in range(3):
        obs_n, reward_n, done_n, info = env.step(action_n)
        ep_reward += sum(reward_n)
        if info['rounds'] != round_idx:
            records.append(reward_n)
            round_idx += 1
        time.sleep(0.03)
        env.render()
env.close()

# plot the cumulative reward curve of player1 against player2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

records = np.array(records)
player1 = np.cumsum(records[:, 0])
player2 = np.cumsum(records[:, 1])
df = pd.DataFrame({'player1': player1, 'player2': player2})
sns.lineplot(data=df)
plt.xlabel('rounds')
plt.ylabel('cumulative reward')
plt.title('Cumulative Reward Curve')
plt.show()
# save the plot
plt.savefig('cumulative_reward_curve.png', dpi=300, bbox_inches='tight')