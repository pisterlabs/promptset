# a manager class that can
# load an autogen flow run an autogen flow and return the response to the client


from typing import Dict
import autogen
import openai
from dotenv import load_dotenv
import os
from src.autogen.agentchat import AssistantAgent, UserProxyAgent

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("API Key not found!")

openai.api_key = api_key


class Manager(object):
    def __init__(self) -> None:

        pass

    def run_flow(self, prompt: str, flow: str = "default") -> None:
        # config_list = autogen.config_list_openai_aoai()
        config_list = [
            {
                'model': 'gpt-3.5-turbo',
                'api_key': api_key,
            }
        ]

        llm_config = {
            "seed": 42,  # seed for caching and reproducibility
            "config_list": config_list,  # a list of OpenAI API configurations
            "temperature": 0,  # temperature for sampling
            "use_cache": True,  # whether to use cache
        }

        assistant = AssistantAgent(
            name="assistant",
            llm_config=llm_config,
        )

        # create a UserProxyAgent instance named "user_proxy"
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="TERMINATE",
            llm_config=llm_config,
            max_consecutive_auto_reply=5,
            is_termination_msg=lambda x: x.get(
                "content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={
                "work_dir": "coding",
                "use_docker": True
            },
            system_message="""Reply TERMINATE if the task has been solved at full
    satisfaction. return to the user for feedback every three messages.
    use a docker container, Otherwise,
    reply CONTINUE, or the reason why the task is not
    solved yet."""
        )

        user_proxy.initiate_chat(
            assistant,
            message=prompt,
        )

        messages = user_proxy.chat_messages[assistant]
        return messages
