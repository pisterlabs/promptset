import os
from dotenv import load_dotenv
import openai
from ..autogen import AssistantAgent, UserProxyAgent

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("API Key not found!")

openai.api_key = api_key

config_list = [
    {
        'model': 'gpt-4',
        'api_key': api_key,
    }
]

# config_list = [{'api_base': 'http://localhost:1234/v1'}]

llm_config = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

code_execution_config = {"work_dir": "web"}

# create an AssistantAgent instance named "assistant"
assistant = AssistantAgent(
    name="assistant",
    llm_config=config_list,
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: x.get(
        "content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "web",
                           "use_docker": True
                           },
    llm_config=config_list,
    system_message="""Reply TERMINATE if the task has been solved at full
    satisfaction. return to the user for feedback every three messages.
    use a docker container, Otherwise,
    reply CONTINUE, or the reason why the task is not
    solved yet."""
)
