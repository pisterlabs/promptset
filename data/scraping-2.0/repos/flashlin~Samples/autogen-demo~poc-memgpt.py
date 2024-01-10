import os

import autogen
import memgpt.autogen.interface as autogen_interface
import openai
from memgpt.persistence_manager import InMemoryStateManager, InMemoryStateManagerWithPreloadedArchivalMemory, InMemoryStateManagerWithEmbeddings, InMemoryStateManagerWithFaiss
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config

MODEL_NAME = "vicuna-7b-v1.5"

os.environ['OPENAI_API_BASE'] = "http://localhost:8000/v1"
os.environ['OPENAI_API_KEY'] = 'NULL'
openai.api_key = "123NULL"

# interface = autogen_interface.AutoGenInterface()
# persistence_manager=InMemoryStateManager()
#persona = "I am a 10x engineer, trained in Python. I was the first engineer at Uber."
# human = "Im a team manager at this company"

config_list = [
    {
        "model": "default",
        "api_type": "NULL",
        "base_base": "http://localhost:8000/v1",
        "api_key": "NULL"
    }
]

llm_config = {"config_list": config_list, "seed": 42}

# memgpt_agent = presets.use_preset(presets.DEFAULT_PRESET,
#                                   model=MODEL_NAME,
#                                   persona=persona,
#                                   human=human,
#                                   interface=interface,
#                                   persistence_manager=persistence_manager,
#                                   agent_config=llm_config)
# bot = memgpt_autogen.MemGPTAgent(
#         name="MemGPT_coder",
#         agent=memgpt_agent,
#     )

# pip install "pyautogen[teachable]"
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "tmp"
    },
    # human_input_mode="TERMINATE",  # needed?
    default_auto_reply="You are going to figure all out by your own. "
    "Work by yourself, the user won't reply until you output `TERMINATE` to end the conversation.",
)



mem_bot = create_memgpt_autogen_agent_from_config(
    name="memgpt_coder",
    llm_config=llm_config,
    system_message="You are a python developer"
)

user_proxy.initiate_chat(
    mem_bot,
    message="Write a Function to print Numbers 1 to 10"
    )
