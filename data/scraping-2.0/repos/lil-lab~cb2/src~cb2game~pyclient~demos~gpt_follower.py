import concurrent.futures
import functools
import logging
import pathlib
from datetime import timedelta
from time import sleep

import fire
import openai

from cb2game.pyclient.client_utils import DescribeMap, FollowerSystemPrompt
from cb2game.pyclient.game_endpoint import Action
from cb2game.pyclient.remote_client import RemoteClient
from cb2game.server.messages.prop import PropUpdate

logger = logging.getLogger(__name__)


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
def call_openai_api_sync(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # model="gpt-4",
        # model="gpt-3.5-turbo",
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


class GPTFollower(object):
    def __init__(self, game_endpoint, pause_per_turn):
        self.instructions_processed = set()
        self.actions = []
        self.game = game_endpoint
        self.exc = None
        self.pause_per_turn = pause_per_turn

    def run(self):
        # Start with the system prompt, explaining the rules of the game.
        game_history = [
            {
                "role": "system",
                "content": FollowerSystemPrompt(),
            },
        ]
        try:
            logger.info(FollowerSystemPrompt())
            game_state = self.game.initial_state()
            (_, _, turn_state, _, _, _) = game_state
            action = Action.NoopAction()
            game_state = self.game.step(action)
            while not self.game.over():
                (mapu, props, turn_state, instrs, actors, feedback) = game_state
                prop_update = PropUpdate(props)
                (leader, follower) = get_actors(game_state)
                description = DescribeMap(
                    mapu, prop_update, instrs, turn_state, follower, leader
                )

                if len(self.actions) == 0:
                    print("===============================")
                    print(description)
                    game_history.append(
                        {
                            "role": "user",
                            "content": description + "\n Enter action: ",
                        }
                    )

                    logger.info(f"========== OPENAI API CALL ==========")
                    response = call_openai_api_sync(messages=game_history)

                    if not response:
                        logger.info(f"step(NOOP)")
                        game_state = self.game.step(Action.NoopAction())
                        continue

                    response_text = response.choices[0].message.content.strip()
                    logger.info(
                        f"############# OPENAI API RESPONSE ##############\n{response_text}\n###############"
                    )
                    sleep(2)
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

                    game_history.append(
                        {
                            "role": "assistant",
                            "content": response_text,
                        }
                    )

                    active_instruction = get_active_instruction(instrs)
                    actions = actions_from_code(action_string, active_instruction.uuid)
                    if len(actions) == 0:
                        # Instead of rapidly polling OpenAI, just quit.
                        logger.info("No actions. Quitting.")
                        break
                    self.actions.extend(actions)
                action = self.actions.pop(0)
                logger.info(f"step({action})")
                game_state = self.game.step(action)
                (_, _, turn_state, _, _, _) = game_state
            print(f"Game over. Score: {turn_state.score}")
        except Exception as e:
            self.exc = e

    def join(self):
        if self.exc:
            raise self.exc


def main(
    host,
    render=False,
    lobby="bot-sandbox",
    pause_per_turn=0,
    api_key="~/openai_api_key.txt",
):
    # Set up OpenAI API key. Expand user directory.
    api_key = pathlib.Path(api_key).expanduser()
    with open(api_key, "r") as f:
        openai.api_key = f.read().strip()

    client = RemoteClient(host, render, lobby_name=lobby)
    connected, reason = client.Connect()
    assert connected, f"Unable to connect: {reason}"

    game, reason = client.JoinGame(
        timeout=timedelta(minutes=5),
        queue_type=RemoteClient.QueueType.FOLLOWER_ONLY,
    )
    assert game is not None, f"Unable to join game: {reason}"

    # Handles game logic.
    follower = GPTFollower(game, pause_per_turn)
    follower.run()
    follower.join()


if __name__ == "__main__":
    fire.Fire(main)
