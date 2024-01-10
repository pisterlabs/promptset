

# Import all the dependencies
import openai
import os
from dotenv import load_dotenv
import autogen
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config # this is the new version of the autogen agent creation function
from memgpt.constants import LLM_MAX_TOKENS
from memgpt.presets.presets import DEFAULT_PRESET

# if you are having problem with SSL use following code
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Load the environment variables and set the openai key parameters
load_dotenv()
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

# Set the openai key
openai.api_key = os.getenv("OPENAI_API_KEY")


config_list_memgpt =  [{
            "model": config_list[0]["model"],
            "context_window": LLM_MAX_TOKENS[config_list[0]["model"]],
            "preset": DEFAULT_PRESET,
            "model_wrapper": None,
            # OpenAI specific
            "model_endpoint_type": "openai",
            "model_endpoint": "https://api.openai.com/v1",
            "openai_key":  os.getenv("OPENAI_API_KEY"),
        }
    ]

llm_config = {'config_list': config_list,"seed":42}

llm_config_memgpt = {"config_list": config_list_memgpt, "seed": 42}

# Create a user agent for interacting with agents
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message="I am human admin",
    human_input_mode="NEVER",
    is_termination_msg=lambda x:x.get("content","").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir":".bot_coding/","last_n_messages": 2,},
    default_auto_reply="...",
)


# Create a manager user agent for interacting with coder agents
pm = autogen.AssistantAgent(
    name="Product_Manager",
    system_message="Creative in software product idea. Don't write any code. Just point out what to do for the coder need to code.",
    llm_config=llm_config)


DEBUG = True
interface_kwargs ={
    "debug":DEBUG,
    "show_inner_thoughts":DEBUG,
    "show_function_outputs":DEBUG,
}

# Create a memgpt coder agent
coder = create_memgpt_autogen_agent_from_config(
        "MemGPT_Coder",
        system_message = "I am 10x software engineer. Trained in python."
                        "I was engineer at Google, Facebook, and Microsoft. I can code anything you want."
                    f"You particular with group chat with a user {{user_proxy.name}}"
                        f"and a product manager {{pm.name}}",
        interface_kwargs=interface_kwargs,
        llm_config =  llm_config_memgpt
        )

# Initializing the group chat between the user and two LLM agents (PM and Coder)
groupchat = autogen.GroupChat(agents=[user_proxy, pm, coder],messages=[],max_round=30)
manger = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

 # Begin the group chat with message from the user
user_proxy.initiate_chat(manger, message="Design ping pong game with score write in python",)