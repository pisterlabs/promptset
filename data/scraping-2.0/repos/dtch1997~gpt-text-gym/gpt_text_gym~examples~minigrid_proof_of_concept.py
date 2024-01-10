import openai
import time
import minigrid  # noqa
import gymnasium as gym
import re
import dotenv

from typing import Dict, List, Tuple, Optional

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball, Box
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from gpt_text_gym import ROOT_DIR

LLM_MODEL = "gpt-4"
OPENAI_TEMPERATURE = 0.0

openai.api_key = dotenv.get_key(ROOT_DIR / ".env", "API_KEY")


class PutNearEnv(MiniGridEnv):

    """
    ## Description

    The agent is instructed through a textual string to pick up an object and
    place it next to another object. This environment is easy to solve with two
    objects, but difficult to solve with more, as it involves both textual
    understanding and spatial reasoning involving multiple objects.

    ## Mission Space

    "put the {move_color} {move_type} near the {target_color} {target_type}"

    {move_color} and {target_color} can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {move_type} and {target_type} Can be "box", "ball" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Drop an object    |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the wrong object.
    2. The agent drop the correct object near the target.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    N: number of objects.

    - `MiniGrid-PutNear-6x6-N2-v0`
    - `MiniGrid-PutNear-8x8-N3-v0`

    """

    def __init__(self, size=6, numObjs=2, max_steps: int | None = None, **kwargs):
        COLOR_NAMES.remove("grey")
        self.size = size
        self.numObjs = numObjs
        self.obj_types = ["key", "ball", "box"]
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[
                COLOR_NAMES,
                self.obj_types,
                COLOR_NAMES,
                self.obj_types,
            ],
        )

        if max_steps is None:
            max_steps = 5 * size

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(
        move_color: str, move_type: str, target_color: str, target_type: str
    ):
        return f"put the {move_color} {move_type} near the {target_color} {target_type}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Types and colors of objects we can generate
        types = ["key", "ball", "box"]

        objs = []
        objPos = []

        def near_obj(env, p1):
            for p2 in objPos:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    return True
            return False

        # Until we have generated all the objects
        while len(objs) < self.numObjs:
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)

            # If this object already exists, try again
            if (objType, objColor) in objs:
                continue

            if objType == "key":
                obj = Key(objColor)
            elif objType == "ball":
                obj = Ball(objColor)
            elif objType == "box":
                obj = Box(objColor)
            else:
                raise ValueError(
                    "{} object type given. Object type can only be of values key, ball and box.".format(
                        objType
                    )
                )

            pos = self.place_obj(obj, reject_fn=near_obj)

            objs.append((objType, objColor))
            objPos.append(pos)

        # Randomize the agent start position and orientation
        self.place_agent()

        # Choose a random object to be moved
        objIdx = self._rand_int(0, len(objs))
        self.move_type, self.moveColor = objs[objIdx]
        self.move_pos = objPos[objIdx]

        # Choose a target object (to put the first object next to)
        while True:
            targetIdx = self._rand_int(0, len(objs))
            if targetIdx != objIdx:
                break
        self.target_type, self.target_color = objs[targetIdx]
        self.target_pos = objPos[targetIdx]

        self.mission = "put the {} {} near the {} {}".format(
            self.moveColor,
            self.move_type,
            self.target_color,
            self.target_type,
        )

    def step(self, action):
        preCarrying = self.carrying

        obs, reward, terminated, truncated, info = super().step(action)

        u, v = self.dir_vec
        ox, oy = (self.agent_pos[0] + u, self.agent_pos[1] + v)
        tx, ty = self.target_pos

        # If we picked up the wrong object, terminate the episode
        if action == self.actions.pickup and self.carrying:
            if (
                self.carrying.type != self.move_type
                or self.carrying.color != self.moveColor
            ):
                terminated = True

        # If successfully dropping an object near the target
        if action == self.actions.drop and preCarrying:
            if self.grid.get(ox, oy) is preCarrying:
                if abs(ox - tx) <= 1 and abs(oy - ty) <= 1:
                    reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info


def limit_tokens_from_string(string: str, model: str, limit: int) -> str:
    """Limits the string to a number of tokens (estimated)."""

    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.encoding_for_model("gpt2")  # Fallback for others.

    encoded = encoding.encode(string)

    return encoding.decode(encoded[:limit])


def openai_call(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 100,
):
    while True:
        try:
            trimmed_prompt = prompt
            # TODO: Enable trimmed prompt.
            # Use 4000 instead of the real limit (4097) to give a bit of wiggle room for the encoding of roles.
            # TODO: different limits for different models.
            # trimmed_prompt = limit_tokens_from_string(prompt, model, 4000 - max_tokens)

            # Use chat completion API
            messages = [{"role": "system", "content": trimmed_prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )
            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break


def get_objects(env: gym.Env) -> List[str]:

    env_str = str(env.unwrapped)

    objects = []
    OBJECT_TO_STR = {
        "wall": "W",
        "floor": "F",
        "door": "D",
        "key": "K",
        "ball": "A",
        "box": "B",
        "goal": "G",
        "lava": "V",
    }
    STR_TO_OBJECT = {v: k for k, v in OBJECT_TO_STR.items()}

    # Map agent's direction to short string
    AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
    STR_TO_AGENT_DIR = {v: k for k, v in AGENT_DIR_TO_STR.items()}

    # Map of colors to short string
    COLOR_TO_STR = {
        "red": "R",
        "green": "G",
        "blue": "B",
        "purple": "P",
        "yellow": "Y",
    }
    STR_TO_COLOR = {v: k for k, v in COLOR_TO_STR.items()}

    rows = env_str.split("\n")
    n_rows = 6
    n_cols = 6
    for row in range(n_rows):
        for col in range(n_cols):
            cell = rows[row][2 * col : 2 * col + 2]
            if cell == "  ":
                # empty cell
                continue
            elif cell[0] in STR_TO_AGENT_DIR and cell[0] == cell[1]:
                # agent
                continue
            elif cell[0] in ("W", "F", "V"):
                # wall, floor, or lava

                # Skip for now
                continue
                # object_name = STR_TO_OBJECT[cell[0]]
            else:
                # interactable object
                object_type = STR_TO_OBJECT[cell[0]]
                object_color = STR_TO_COLOR[cell[1]]
                object_name = f"{object_color} {object_type}"
            objects.append(object_name)

    return objects


def get_objects_in_view(obs: Dict) -> List[str]:
    """
    Get objects in the agent's field of view.
    """
    pass


def get_inventory(env: gym.Env) -> str:
    object = env.unwrapped.carrying
    if object is None:
        return "nothing"
    else:
        return f"{object.color} {object.type}"


def describe_environment(env: gym.Env, obs: Dict) -> str:
    objects = get_objects(env)
    inventory = get_inventory(env)

    # TODO: Only get visible objects
    env_description = f"""
You are in a room. 
You see: {', '.join(objects)}.
You are facing: {obs["direction"]}.
You are currently holding: {inventory}.
"""
    return env_description


def planning_agent(env, obs, previous_goal: str) -> str:
    prompt = f"""
You are controlling a simulated agent to complete tasks. 
The overall goal is: {obs["mission"]}.
The previous goal was: {previous_goal}. 

{describe_environment(env, obs)}

Describe the next goal in one sentence. Be concise.
"""
    print(f"\n****PLANNING AGENT PROMPT****\n{prompt}\n")
    response = openai_call(prompt)
    print(f"\n****PLANNING AGENT RESPONSE****\n{response}\n")
    return response.strip().lower()


def evaluation_agent(env, obs, current_goal: str):
    prompt = f"""
You are controlling a simulated agent to complete tasks. 
The overall goal is: {obs["mission"]}.
The current goal is: {current_goal}. 
        
{describe_environment(env, obs)}

Has the current goal been reached? Answer yes or no.
"""
    print(f"\n****EVALUATION AGENT PROMPT****\n{prompt}\n")
    response = openai_call(prompt)
    print(f"\n****EVALUATION AGENT RESPONSE****\n{response}\n")
    return response.strip().lower()


from minigrid.core.actions import Actions


def key_handler(event, env) -> Optional[Actions]:
    key: str = event.key
    print("pressed", key)

    if key == "escape":
        env.close()
        return
    if key == "backspace":
        env.reset()
        return

    key_to_action = {
        "left": Actions.left,
        "right": Actions.right,
        "up": Actions.forward,
        "space": Actions.toggle,
        "pageup": Actions.pickup,
        "pagedown": Actions.drop,
        "tab": Actions.pickup,
        "left shift": Actions.drop,
        "enter": Actions.done,
    }
    return key_to_action.get(key)


def manual_control():
    import pygame

    env = PutNearEnv(size=6, numObjs=2, max_steps=50, render_mode="human")
    obs, _ = env.reset()
    env.render()
    previous_goal = ""
    current_goal = planning_agent(env, obs, previous_goal)

    while True:
        # Step the agent
        # TODO: Implement CLI for manual control
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                event.key = pygame.key.name(int(event.key))
                action = key_handler(event, env)
                if action is None:
                    continue
                obs, _, terminated, truncated, _ = env.step(action)
                env.render()

                # Evaluate the agent
                evaluation = evaluation_agent(env, obs, current_goal)
                if evaluation == "yes":
                    previous_goal = current_goal
                    current_goal = planning_agent(env, obs, previous_goal)
                elif evaluation == "no":
                    pass
                else:
                    raise ValueError(f"Invalid evaluation: {evaluation}")


def main():
    env = PutNearEnv(size=6, numObjs=2, max_steps=50, render_mode="human")
    obs, _ = env.reset()
    env.render()
    previous_goal = ""
    current_goal = planning_agent(env, obs, previous_goal)

    while True:
        # Step the agent
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        env.render()

        # Evaluate the agent
        evaluation = evaluation_agent(env, obs, current_goal)
        if evaluation == "yes":
            previous_goal = current_goal
            current_goal = planning_agent(env, obs, previous_goal)
        elif evaluation == "no":
            pass
        else:
            raise ValueError(f"Invalid evaluation: {evaluation}")


if __name__ == "__main__":
    manual_control()
