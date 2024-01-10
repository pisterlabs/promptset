import gymnasium as gym
from minigrid.core.constants import *
import matplotlib.pyplot as plt
import random
import openai
import json
import random

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
DIR_TO_STR = {0: "right", 1: "down", 2: "left", 3: "up"}
ACTION_TO_STR = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up an object",
    4: "drop the object",
    5: "toggle/activate an object",
    6: "finish the environment"
}


def img_to_str(img):
    """
    IDX_TO_OBJECT, IDX_TO_COLOR, IDX_TO_STATE mappings see:
    https://huggingface.co/spaces/flowers-team/SocialAIDemo/blob/b5027a4ec69027c0ae6a6c471316ca5cb1c36560/gym-minigrid/gym_minigrid/minigrid.py
    """
    result = ""
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            obj = IDX_TO_OBJECT[img[i, j, 0]]
            color = IDX_TO_COLOR[img[i, j, 1]]
            state = IDX_TO_STATE[img[i, j, 2]]
            if obj in ["unseen", "empty", "wall"]:
                result += f"{obj}, "
            elif obj in ["ball", "box"]:
                result += f"{color} {obj}, "
            else:
                result += f"{color} {state} {obj}, "
        result += '\n'
    return result


class LLMRewardFunction():

    def __init__(self, query_interval=10000, decay=0.7, llm_temperature=0.3, llm='gpt-3.5-turbo'):
        self.query_interval = query_interval
        self.trajectory = []
        self.steps_since_last_query = 0
        self.decay = decay
        self.decay_rate = decay
        self.steps_before_first_query = 0
        self.llm_temperature = llm_temperature
        self.steps_start_query = random.randint(0, query_interval // 2)
        self.llm = llm

    def reshape_reward(self, observation, action, reward, done):
        # If it's time to query GPT or the trajectory is empty
        if self.steps_since_last_query >= self.query_interval and not self.trajectory and self.steps_before_first_query >= self.steps_start_query:
            self.trajectory = self.get_gpt_trajectory(observation)
            self.steps_since_last_query = 0
            self.decay = self.decay_rate

        # Check if the agent's action aligns with the expected trajectory
        if self.trajectory:
            expected_action = self.trajectory.pop(0)
            if action.item() == expected_action:
                shaped_reward = reward + self.decay  # Positive reward for following the trajectory
                self.decay *= self.decay
            else:
                shaped_reward = reward  # Neutral or negative reward for deviating
        else:
            shaped_reward = reward

        self.steps_since_last_query += 1
        self.steps_before_first_query += 1
        return shaped_reward

    def get_gpt_trajectory(self, obs):
        trajectory_json = self.trajectory_gen(obs)
        if trajectory_json.startswith("'") and trajectory_json.endswith("'"):
            trajectory_json = trajectory_json[1:-1]
        trajectory_str = json.loads(trajectory_json)
        # Convert the trajectory string to a list of actions
        trajectory = [trajectory_str[f'{i}'] for i in range(10)]
        return trajectory

    def trajectory_gen(self, obs):
        prompt = self.get_prompt_str(obs)
        output = openai.ChatCompletion.create(
            model=self.llm,
            messages=[
                {
                    "role":
                        "system",
                    "content":
                        '''You are now a route planning assistant in the Minigrid game. Each action is represented by an integer from 0 to 6, and here're the actions you may consider:
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up the object",
    4: "drop the object",
    5: "toggle the object",
    6: "finish the environment"
    Your answer should only be in json format, no analysis is needed. Each action is represented by an integer from 0 to 6. The format should be in json, and it should be like:
    '{"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 0, "8": 1, "9": 2}'
    '''
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=0.3,
            timeout=10,
            # if strictly follow the format, the max tokens should be 61
            max_tokens=70,
        )
        return output.choices[0].message['content']

    def get_prompt_str(self, obs):
        image, direction, mission = img_to_str(obs['image']), DIR_TO_STR[obs['direction']], obs['mission']
        return f'''Your mission is to {mission}. You're currently facing {direction}. You can only see what's in front of you. The unseen region are blocked by objects or walls. Assume you are at the center of the last row. The first row is the furthest from you. Here's what you see:
    {image}
    What are the best actions to take in the next 10 steps to achieve your mission?
    '''


# The things below are just test code.
if __name__ == "__main__":
    env = gym.make("BabyAI-GoToImpUnlock-v0", render_mode='rgb_array')
    # Reset the environment to get the initial state
    obs = env.reset()
    # Take some actions and continue displaying the state
    for _ in range():
        action = env.action_space.sample()  # Replace with your desired action
        obs, reward, terminated, truncated, info = env.step(action)
        plt.figure()
        plt.imshow(env.render())
        plt.savefig("test.png")
