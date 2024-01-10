from model_config import get_config
from llama_cpp import Llama
from openai_api import start_openai_api_thread

from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.teachable_agent import TeachableAgent

import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = get_config("OpenHermes-2.5-neural-chat-7B-v3-1-7B")
#config = get_config("Mistral-7B-Instruct-v0.1-function-calling-v2")

start_openai_api_thread(Llama(**config))

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from termcolor import colored
except ImportError:
    def colored(x, *args, **kwargs):
        return x


verbosity = 0  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
recall_threshold = 1.5  # Higher numbers allow more (but less relevant) memos to be recalled.
cache_seed = None  # Use an int to seed the response cache. Use None to disable caching.


def create_teachable_agent(reset_db=False):
    """Instantiates a TeachableAgent using the settings from the top of this file."""
    # Load LLM inference endpoints from an env variable or a file
    # See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
    # and OAI_CONFIG_LIST_sample
    config_list = [{
        "base_url": "http://localhost:8000/v1",
        "model": "gpt-4",
        "api_key": "dontcare",
    }]

    teachable_agent = TeachableAgent(
        name="teachableagent",
        llm_config={"config_list": config_list, "timeout": 120, "cache_seed": cache_seed},
        teach_config={
            "verbosity": verbosity,
            "reset_db": reset_db,
            "path_to_db_dir": "./tmp/interactive/teachable_agent_db",
            "recall_threshold": recall_threshold,
        },
    )
    return teachable_agent


def interact_freely_with_user():
    """Starts a free-form chat between the user and TeachableAgent."""

    # Create the agents.
    print(colored("\nLoading previous memory (if any) from disk.", "light_cyan"))
    teachable_agent = create_teachable_agent(reset_db=False)
    user = UserProxyAgent("user", human_input_mode="ALWAYS")

    # Start the chat.
    teachable_agent.initiate_chat(user, message="Greetings, I'm a teachable user assistant! What's on your mind today?")

    # Let the teachable agent remember things that should be learned from this chat.
    teachable_agent.learn_from_user_feedback()

    # Wrap up.
    teachable_agent.close_db()


if __name__ == "__main__":
    """Lets the user test TeachableAgent interactively."""
    interact_freely_with_user()