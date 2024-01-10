import autogen
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    file_location="."
)
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding"})
user_proxy.initiate_chat(assistant, message="create a txt file containing a small poem about a dog named Raki.")
# This initiates an automated chat between the two agents to solve the task