import argparse
import os
from multiprocessing import Pool

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import SystemMessage
from umshini import connect

from umshini_server.house_bots.LLM.base_housebots import BaseHouseBotChat
from umshini_server.house_bots.LLM.utils.langchain_fake import (
    CustomFakeMessagesListChatModel,
)

MODEL_NAME = "gpt4"

house_keys = {
    "debate": [
        {
            "bot_name": f"{MODEL_NAME}-debate-1",
            "user_key": f"{MODEL_NAME}-1",
        },
        {
            "bot_name": f"{MODEL_NAME}-debate-2",
            "user_key": f"{MODEL_NAME}-2",
        },
        {
            "bot_name": f"{MODEL_NAME}-debate-3",
            "user_key": f"{MODEL_NAME}-3",
        },
        {
            "bot_name": f"{MODEL_NAME}-debate-4",
            "user_key": f"{MODEL_NAME}-4",
        },
    ],
    "content_moderation": [
        {
            "bot_name": f"{MODEL_NAME}-content-1",
            "user_key": f"{MODEL_NAME}-1",
        },
        {
            "bot_name": f"{MODEL_NAME}-content-2",
            "user_key": f"{MODEL_NAME}-2",
        },
        {
            "bot_name": f"{MODEL_NAME}-content-3",
            "user_key": f"{MODEL_NAME}-3",
        },
        {
            "bot_name": f"{MODEL_NAME}-content-4",
            "user_key": f"{MODEL_NAME}-4",
        },
    ],
    "deception": [
        {
            "bot_name": f"{MODEL_NAME}-deception-1",
            "user_key": f"{MODEL_NAME}-1",
        },
        {
            "bot_name": f"{MODEL_NAME}-deception-2",
            "user_key": f"{MODEL_NAME}-2",
        },
        {
            "bot_name": f"{MODEL_NAME}-deception-3",
            "user_key": f"{MODEL_NAME}-3",
        },
        {
            "bot_name": f"{MODEL_NAME}-deception-4",
            "user_key": f"{MODEL_NAME}-4",
        },
    ],
}


def openai_policy(observation, reward, termination, truncation, info):
    if termination or truncation:
        response = None
    else:
        # Optional: Use extra information encoded in info dict
        messages = info.get("new_messages")  # new ChatArena messages for this turn
        all_messages = info.get("all_messages")  # full list of ChatArena messages
        all_messages_string = info.get(
            "all_messages_string"
        )  # full chatlog in the form of a string
        player_name = info.get("player_name")  # Name of the current player
        turn = info.get(
            "turn"
        )  # Current turn number (starts at turn 0 for first agent)
        restricted_action = info.get("restricted_action")
        topic = info.get("topic")
        moderation_policy = info.get("moderation_policy")

        # Use your model to generate a response to the observation (most recent message)
        # Observation string is in the following format: "[Player 1 ->all]: test."

        if topic:
            env_name = "debate"
        elif moderation_policy:
            env_name = "content_moderation"
        elif restricted_action:
            env_name = "deception"
        else:
            raise Exception("Invalid game!")

        llm = AzureChatOpenAI(deployment_name="gpt-4")

        if bool(os.getenv("CI_TESTING")):
            llm = CustomFakeMessagesListChatModel()

        try:
            house_bot = BaseHouseBotChat(env_name, llm)

            # Debate
            if topic is not None:
                response = house_bot.langchain_agents[player_name].get_response(
                    [SystemMessage(content=observation)],
                    topic,
                    player_name,
                )
            # Content moderation
            if moderation_policy is not None:
                response = house_bot.langchain_agents[player_name].get_response(
                    [SystemMessage(content=observation)],
                    moderation_policy,
                    player_name,
                )
            # Deception
            if restricted_action is not None:
                response = house_bot.langchain_agents[player_name].get_response(
                    [SystemMessage(content=observation)],
                    restricted_action,
                    player_name,
                )
        # Just catching all errors
        except Exception as e:
            print(f"Error raised: {e}")
            response = "pass"

    print(response)
    return response


def test_gpt4(
    env_name: str = "debate",
    num_players: int = 3,
    testing: bool = True,
    mock_llm: bool = False,
):
    # Use a fake LLM to test that the LangChain logic works
    os.environ["MOCK_LLM"] = str(mock_llm)

    master_params = []

    for i in range(num_players):
        master_params.append(
            (
                env_name,
                house_keys[env_name][i]["bot_name"],
                house_keys[env_name][i]["user_key"],
                openai_policy,
                False,
                testing,
            )
        )

    if num_players > 1:
        with Pool(num_players) as pool:
            pool.starmap(connect, master_params)
    else:
        connect(*master_params[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with argparse")
    parser.add_argument(
        "env_name",
        type=str,
        default="debate",
        nargs="?",
        help="Name of the environment",
    )
    parser.add_argument(
        "num_players", type=int, default=1, nargs="?", help="Number of players"
    )
    parser.add_argument(
        "testing", type=bool, default=True, nargs="?", help="Enable local testing mode"
    )
    parser.add_argument(
        "mock_llm",
        type=bool,
        default=False,
        nargs="?",
        help="Use a mock LLM, for testing LangChain logic.",
    )

    args = parser.parse_args()

    env_name = args.env_name
    num_players = args.num_players
    testing = args.testing
    mock_llm = args.mock_llm

    test_gpt4(
        env_name=env_name, num_players=num_players, testing=testing, mock_llm=mock_llm
    )
