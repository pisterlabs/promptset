# Auto Generated Agent Chat: Group Chat

import autogen
import openai
from decouple import config


openai.api_key = config("OPENAI_API_KEY")


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

user_proxy = autogen.UserProxyAgent(
    name="admin_user_proxy",
    system_message="You are a human admin",
    human_input_mode="TERMINATE",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    },
)

# Find a latest paper about gpt-4 on arxiv and find its potential applicat

coder = autogen.AssistantAgent(
    name="coder",
    llm_config={
        "config_list": config_list,
        "temperature": 0
    }
)

pm = autogen.AssistantAgent(
    name="Product manager",
    system_message="You are a creative product manager, who gives creative software ideas.",
    llm_config={
        "config_list": config_list,
        "temperature": 0
    }
)

group_chat = autogen.GroupChat(agents=[pm, user_proxy, coder], messages=[
], max_round=12, admin_name="Admin")

manager = autogen.GroupChatManager(groupchat=group_chat, name="chat_manager")

user_proxy.initiate_chat(
    manager,
    message="Find the latest news on GPTs and how they can be used to develop software, I also want you to come up with an idea of how we can use this to develop a custom software. I want you to write the final report ina pdf file called 'report.pdf'"
)
