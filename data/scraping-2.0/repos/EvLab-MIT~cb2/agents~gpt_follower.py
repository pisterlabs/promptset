import concurrent.futures
import functools
import logging
from dataclasses import dataclass

import openai
from mashumaro.mixins.json import DataClassJSONMixin

from agents.agent import Agent, Role
from py_client.client_utils import (
    DescribeMap,
    FollowerSystemPrompt,
    SingleActionSystemPrompt,
)
from py_client.game_endpoint import Action, GameState
from server.messages.prop import PropUpdate

logger = logging.getLogger(__name__)


@dataclass
class GPTFollowerConfig(DataClassJSONMixin):
    """Configuration for initializing a GPTFollower agent.

    For help choosing a value for `model`, see:
        https://platform.openai.com/docs/models/overview
    To get an API key, see:
        https://platform.openai.com/account/api-keys
    """

    gpt_api_key: str
    queueing_enabled: bool = False
    model: str = "gpt-3.5-turbo"


class GPTFollower(Agent):
    def __init__(self, config: GPTFollowerConfig):
        self.queueing_enabled = config.queueing_enabled
        self.gpt_api_key = config.gpt_api_key
        self.model = config.model
        game_explanation = (
            FollowerSystemPrompt()
            if self.queueing_enabled
            else SingleActionSystemPrompt()
        )
        self.game_history = [
            {"role": "system", "content": game_explanation},
        ]
        self.action_queue = []

    # OVERRIDES role
    def role(self) -> Role:
        return Role.FOLLOWER

    # OVERRIDES choose_action
    def choose_action(self, game_state: GameState, action_mask=None) -> Action:
        """Chooses an action to take, given a game state.

        If queueing_enabled is true in the constructor, one prompt can
        return multiple instructions (to save on GPT API calls). They are
        queued up in self.action_queue.  If the queue is empty, then we need
        to call the GPT API to get more actions.

        Action masking not supported. This parameter is ignored.

        If queueing is not enabled, each prompt returns one instruction.
        """
        if len(self.action_queue) > 0 and self.queueing_enabled:
            return self.action_queue.pop(0)
        # Fetch more actions.
        [mapu, props, turn_state, instrs, _, _] = game_state
        prop_update = PropUpdate(props)
        (leader, follower) = get_actors(game_state)
        description = DescribeMap(
            mapu, prop_update, instrs, turn_state, follower, leader
        )
        self.game_history.append(
            {
                "role": "user",
                "content": description,
            }
        )
        response = call_openai_api_sync(messages=self.game_history)

        if not response:
            return Action.NoopAction()

        response_text = response.choices[0].message.content.strip()
        action_string = ""
        # Split by lines. If a line starts with "THOUGHTS:" or "THOUGHT:", then
        # print the line. If a line starts with "ACTIONS:" or "ACTION:", then
        # collect everything after the colon and split by comma.
        lines = response_text.split("\n")
        for line in lines:
            if line.startswith("THOUGHTS:") or line.startswith("THOUGHT:"):
                print(f"GPT thought: `{line}`")
            elif line.startswith("ACTIONS:") or line.startswith("ACTION:"):
                action_string = line.split(":")[1]
                break
        self.game_history.append(
            {
                "role": "assistant",
                "content": response_text,
            }
        )
        active_instruction = get_active_instruction(instrs)
        actions = actions_from_code(action_string, active_instruction.uuid)
        if len(actions) == 0:
            return Action.NoopAction()

        if self.queueing_enabled:
            self.action_queue = actions[1:]
            return actions[0]

        return actions[0]


def timeout_decorator(timeout):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout)
                except concurrent.futures.TimeoutError:
                    print("Function call timed out")

        return wrapper

    return decorator


@timeout_decorator(timeout=20)
def call_openai_api_sync(messages, model="gpt-3.5-turbo"):
    """Calls OpenAI API synchronously with a timeout. Some other values for model parameter:"""
    # model="gpt-4",
    # model="gpt-3.5-turbo",
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=50,
        n=1,
        temperature=0.5,
    )
    return response


def actions_from_code(action_code, i_uuid: str = None):
    # Split action code by comma.
    characters_in_prompt = action_code.split(",")
    if len(characters_in_prompt) == 0:
        logger.warning("Empty action string.")
        return None
    actions = []
    for c in characters_in_prompt:
        # Convert to lower and strip whitespace.
        c = c.lower().strip()
        logger.info(f"Action code: `{c}`")
        if "forward".startswith(c):
            actions.append(Action.Forwards())
        elif "backward".startswith(c):
            actions.append(Action.Backwards())
        elif "left".startswith(c):
            actions.append(Action.Left())
        elif "right".startswith(c):
            actions.append(Action.Right())
        elif "done".startswith(c):
            actions.append(Action.InstructionDone(i_uuid))
        else:
            logger.warning(f"Invalid action code: {c}")
    return actions


def get_active_instruction(instructions):
    for instruction in instructions:
        if not instruction.completed and not instruction.cancelled:
            return instruction
    return None


def get_actors(game_state):
    (
        _,
        _,
        _,
        _,
        actors,
        _,
    ) = game_state
    if len(actors) == 1:
        return (None, actors[0])
    else:
        return actors
