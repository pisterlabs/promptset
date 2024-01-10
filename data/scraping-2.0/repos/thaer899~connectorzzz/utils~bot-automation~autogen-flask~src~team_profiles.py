import autogen
from src.autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from src.autogen.agentchat import AssistantAgent, UserProxyAgent
import chromadb
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("API Key not found!")

openai.api_key = api_key


class TeamProfiles:
    def __init__(self):
        # Configurations for the LLM (Language Model)
        self.llm_config = self.get_llm_config(api_key)

    @staticmethod
    def termination_msg(x): return isinstance(
        x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

    @staticmethod
    def get_llm_config(api_key: str):
        return {
            "request_timeout": 60,
            "seed": 42,
            "config_list": [{'model': 'gpt-4', 'api_key': api_key}],
            "temperature": 0,
        }

    def create_boss(self):
        return AssistantAgent(
            name="Boss",
            is_termination_msg=self.termination_msg,
            system_message="An experienced tech leader with a vision to drive innovation and excellence. Creative in software product ideas, with a deep understanding of market needs",
            human_input_mode="ALWAYS",
            llm_config=self.llm_config,
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": "coding/teamchat",
                "use_docker": True},

        )

    def create_aid(self):
        return AssistantAgent(
            name="Boss_Assistant",
            is_termination_msg=self.termination_msg,
            system_message="A reliable and knowledgeable aid with a knack for problem-solving. Detail-oriented reviewer ensuring code quality and functionality.",
            llm_config=self.llm_config
        )

    def create_coder(self):
        return AssistantAgent(
            name="Senior_Python_Engineer",
            is_termination_msg=self.termination_msg,
            system_message="A highly skilled coder with the ability to translate complex problems into executable code.",
            llm_config=self.llm_config,
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": "coding/teamchat",
                "use_docker": True}
        )
