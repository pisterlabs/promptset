import os
from autogen import config_list_from_json
from dotenv import load_dotenv
import autogen
import openai
import sys

sys.path.insert(0, '/src/useCases/openAI/agent/auto_agent')

from group_chat.research import research
from group_chat.content_writer import write_content

load_dotenv()
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_config_content_assistant = {
    "functions": [
        {
            "name": "research",
            "description": "research about a given topic, return the research material including reference links",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The topic to be researched about",
                        }
                    },
                "required": ["query"],
            },
        },
        {
            "name": "write_content",
            "description": "Write content based on the given research material & topic",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "research_material": {
                            "type": "string",
                            "description": "research material of a given topic, including reference links when available",
                        },
                        "topic": {
                            "type": "string",
                            "description": "The topic of the content",
                        }
                    },
                "required": ["research_material", "topic"],
            },
        },
    ],
    "config_list": config_list
}

writing_assistant = autogen.AssistantAgent(
    name="writing_assistant",
    system_message="You are a writing assistant, you can use research function to collect latest information about a given topic, and then use write_content function to write a very well written content; Always reply TERMINATE when your task is done, return the final result as the last message;when the task is done",
    llm_config=llm_config_content_assistant,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    function_map={
        "research": research,
        "write_content": write_content,
    }
)

user_proxy.initiate_chat(writing_assistant, message="viết blog về chuyến du lịch đến Thành phố Đà Lạt")