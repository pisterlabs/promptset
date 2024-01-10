import os
import autogen
import memgpt.autogen.memgpt_agent as memgpt_autogen
import memgpt.autogen.interface as autogen_interface
import memgpt.agent as agent
import memgpt.system as system
import memgpt.utils as utils
import memgpt.constants as constants
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
from memgpt.persistence_manager import (
    InMemoryStateManager,
    InMemoryStateManagerWithPreloadedArchivalMemory,
    InMemoryStateManagerWithFaiss,
)
from memgpt.autogen.memgpt_agent import create_autogen_memgpt_agent
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

config_list = [
    {"model": "gpt-4"},
]


# If USE_MEMGPT is False, then this example will be the same as the official AutoGen repo (https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb)
# If USE_MEMGPT is True, then we swap out the "coder" agent with a MemGPT agent
USE_MEMGPT = True
# If DEBUG is False, a lot of MemGPT's inner workings output is suppressed and only the final send_message is displayed.
# If DEBUG is True, then all of MemGPT's inner workings (function calls, etc.) will be output.
DEBUG = False


llm_config = {"config_list": config_list, "seed": 42}
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
)

interface = autogen_interface.AutoGenInterface()  # how MemGPT talks to AutoGen
persistence_manager = InMemoryStateManager()
persona = "I'm a 10x engineer at a FAANG tech company."
human = "I'm a team manager at a FAANG tech company."
# memgpt_agent = presets.use_preset(
#     presets.DEFAULT_PRESET,
#     llm_config,
#     "gpt-4",
#     persona,
#     human,
#     interface,
#     persistence_manager,
# )

# non-MemGPT PM
pm = autogen.AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas.",
    llm_config=llm_config,
)


# MemGPT coder
# coder = memgpt_autogen.MemGPTAgent(
#     name="MemGPT_coder",
#     agent=memgpt_agent,
# )
coder = create_autogen_memgpt_agent(
    "MemGPT_coder",
    persona_description="I am a 10x engineer, trained in Python. I was the first engineer at Uber (which I make sure to tell everyone I work with).",
    user_description=f"You are participating in a group chat with a user ({user_proxy.name}) and a product manager ({pm.name}).",
    interface_kwargs={"debug": DEBUG},
)


groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(manager, message="First send the message 'Let's go Mario!'")
