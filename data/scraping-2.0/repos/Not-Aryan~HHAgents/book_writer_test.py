
import autogen 
import openai
import os


openai.api_key = os.getenv("OPENAI_API_KEY") or "YOUR KEY HERE"

config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
    "model": ["gpt-4"],
    },
)

gpt4_config = {
    "seed": 42, # change the seed for different trials
    "temperature": 0,
    "config_list": config_list_gpt4,
    "request_timeout": 120,
}

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=gpt4_config,
    system_message='''Engineer. You follow an approved plan. 
    You write python/shell code to solve tasks. 
    Wrap the code in a code block that specifies the script type. 
    The user can't modify your code. So do not suggest incomplete code which requires others to modify. 
    Don't use a code block if it's not intended to be executed by the executor.
     Don't include multiple code blocks in one response.
      Do not ask others to copy and paste the result. Check the execution'''
)

