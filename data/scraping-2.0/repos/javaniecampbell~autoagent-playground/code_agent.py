from autogen import  config_list_from_json
import autogen
# import openai
# openai.api_key = '<OPENAI_KEY_HERE>'
config_list = config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-4",  "gpt-4-0314",  "gpt4",  "gpt-4-32k",  "gpt-4-32k-0314",  "gpt-4-32k-v0314"], 
            # "model": ["gpt-3.5-turbo", "gpt-3.5-turbo-0613",  "gpt3.5-turbo",  "gpt-3.5-turbo-16k",  "gpt-3.5-turbo-16k-0613",  "gpt-3.5-turbo-16k-v0613"], 
        }
)
llm_config={ 
    "config_list": config_list, 
    "seed": 42, 
    "request_timeout": 120
}

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message="A human admin who will give the idea and run the code provided by the Coder.",
    code_execution_config={"last_n_messages":2,"work_dir":"groupchat_coding"},
    human_input_mode="ALWAYS"
)

coder = autogen.AssistantAgent(
    name="Coder", 
    llm_config=llm_config
)

pm = autogen.AssistantAgent(
    name="product_manager", 
    system_message="You will help break down the initial idea into a well scoped requirements for the coder; Do NOT get involved in future conversations or error fixing.",
    llm_config=llm_config
)

# Create a group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy,coder,pm],
    messages=[]
)
manager = autogen.GroupChatManager(groupchat=groupchat,llm_config=llm_config)

user_proxy.initiate_chat(
    manager,
    message="Build a classic & basic pong game with 2 players in python"
)
