import threading
from .agents.user_proxy_agent import UserProxyAgent
from .agents.user_proxy_web_agent import UserProxyWebAgent
from .agents.assistant_agent import AssistantAgent
from .agents.groupchat import GroupChat, GroupChatManager
import queue
import openai
from dotenv import load_dotenv
import os
from typing import Dict

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("API Key not found!")

openai.api_key = api_key


class TeamProfiles:
    def __init__(self, chat_id=None, websocket=None):
        # Configurations for the LLM (Language Model)
        self.llm_config = self.get_llm_config(api_key)
        self.websocket = websocket
        self.chat_id = chat_id
        self.client_sent_queue = queue.LifoQueue()
        self.client_receive_queue = queue.LifoQueue()
        self.queue_event = threading.Event()
        self.boss = self.create_boss()
        self.aid = self.create_aid()
        self.coder = self.create_coder()
        self.new_data_event = threading.Event()
        self.processing_done_event = threading.Event()
        self.new_reply_event = threading.Event()

    @staticmethod
    def termination_msg(x): return isinstance(
        x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

    @staticmethod
    def get_llm_config(api_key: str):
        return {
            "request_timeout": 60,
            "seed": 42,
            "config_list": [{'model': 'gpt-3.5-turbo', 'api_key': api_key}],
            "temperature": 0,
        }

    def create_boss(self):
        boss = UserProxyWebAgent(
            name="Boss",
            is_termination_msg=self.termination_msg,
            system_message="An experienced tech leader with a vision to drive innovation and excellence. Creative in software product ideas, with a deep understanding of market needs",
            human_input_mode="TERMINATE",
            llm_config=self.llm_config,
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": "coding/teamchat",
                "use_docker": True},
            client_sent_queue=self.client_sent_queue,
            client_receive_queue=self.client_receive_queue
        )
        return boss

    def create_aid(self):
        aid = AssistantAgent(
            name="Boss_Assistant",
            is_termination_msg=self.termination_msg,
            system_message="A reliable and knowledgeable aid with a knack for problem-solving. Detail-oriented reviewer ensuring code quality and functionality.",
            llm_config=self.llm_config,
            client_sent_queue=self.client_sent_queue,
            client_receive_queue=self.client_receive_queue
        )
        return aid

    def create_coder(self):
        coder = AssistantAgent(
            name="Senior_Python_Engineer",
            is_termination_msg=self.termination_msg,
            system_message="A highly skilled coder with the ability to translate complex problems into executable code.",
            llm_config=self.llm_config,
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": "coding/teamchat",
                "use_docker": True},
            client_sent_queue=self.client_sent_queue,
            client_receive_queue=self.client_receive_queue
        )
        return coder

    def get_queues(self):
        print("Inside get_queues method...")
        print(self.client_sent_queue)
        print(self.client_receive_queue)
        return self.client_sent_queue, self.client_receive_queue

    def set_thread(self, thread):
        self.thread = thread

    def initiate_team_chat(self, prompt: str, chat_id: str) -> Dict:
        """Initiate a team chat and return the messages"""
        print("Inside initiate_team_chat method...")

        # Resetting and creating new agents
        boss = self.boss
        aid = self.aid
        coder = self.coder

        # Creating a new group chat with the agents
        groupchat = GroupChat(
            agents=[boss, aid, coder],
            messages=[],
            max_round=5,
        )
        manager = GroupChatManager(
            groupchat=groupchat,
            name="pm",
            llm_config=self.llm_config,
            max_consecutive_auto_reply=5,
            client_sent_queue=self.client_sent_queue,
            client_receive_queue=self.client_receive_queue,
        )

        manager.initiate_chat(
            boss,
            clear_history=True,
            message=prompt,
        )

        formatted_messages = [
            {"name": msg["name"], "content": msg["content"]} for msg in groupchat.messages]

        for msg in formatted_messages:
            self.client_receive_queue.put(msg)

        self.new_reply_event.set()

        # return response
        return {"status": True, "message": "Chat initiated successfully."}
