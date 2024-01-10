import sys
import os
from datetime import datetime
import bot_config

sys.path.append("/root/projects")
import common.bot_logging
from common.bot_handler import RabbitHandler
from common.bot_comms import (
    from_bot_manager,
    send_to_user,
    get_input,
    send_prompt,
    from_bot_to_bot_manager,
    send_to_another_bot,
    publish_error,
)
from common.bot_forward import Forward


from loaders.erp_api import ERPGETLIST, ERPGETDOC, ERPGETFIELDS

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools, Tool
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import MessagesPlaceholder


class aiBot:
    def __init__(self):
        # common.bot_logging.bot_logger = common.bot_logging.logging.getLogger('BotInstance')
        # common.bot_logging.bot_logger.addHandler(common.bot_logging.file_handler)
        common.bot_logging.bot_logger.info(f"Init Bot Instance")
        self.credentials = []
        self.initialised = False

    def bot_init(self):
        # Init AI
        if not self.initialised:
            os.environ["OPENAI_API_KEY"] = bot_config.OPENAI_API_KEY
            self.llm = ChatOpenAI(
                model_name=bot_config.MAIN_AI, temperature=0, verbose=True
            )
            self.handler = RabbitHandler()
            self.tools = self.load_tools(self.llm)

            # Initate Agent Executor
            self.chat_history = MessagesPlaceholder(variable_name="chat_history")
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history", k=2, return_messages=True
            )
            self.memory.buffer.clear()

            # self.agent_executor = initialize_agent(
            #     tools=self.tools,
            #     llm=self.llm,
            #     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            #     verbose=True,
            #     max_iterations=5,
            # )
            self.agent_executor = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True,
                max_iterations=5,
                agent_kwargs={
                    "memory_prompts": [self.chat_history],
                    "input_variables": ["input", "agent_scratchpad", "chat_history"],
                },
            )
            self.initialised = True

    # Main Loop
    def process_messages(self):
        "loop"
        message = from_bot_manager()
        if message:
            incoming_credentials = message.get("credentials")
            prompt = message.get("prompt")

            if incoming_credentials:
                self.credentials = incoming_credentials
                bot_config.OPENAI_API_KEY = self.get_credential("openai_api")
                bot_config.FRIENDLY_NAME = self.get_credential("user_name")
                bot_config.ERP_URL = self.get_credential("erp_url")
                bot_config.ERP_API_KEY = self.get_credential("erp_api_key")
                bot_config.ERP_API_SECRET = self.get_credential("erp_api_secret")
                self.bot_init()

            if prompt == "credential_update":
                send_to_user(f"{bot_config.BOT_ID} Settings Updated")
                return

            if prompt == "start_scheduled_tasks":
                # self.pause_schedule = False
                return

            if prompt == "ping":
                send_to_user(f"PID:{os.getpid()} - pong")
                return

            if prompt == "kill":
                send_to_user(f"PID:{os.getpid()} - Shutting Down")
                sys.exit()
                return

            if prompt and self.initialised:
                # AI goes here
                self.process_model(prompt)
                # bot ends itself
                # sys.exit()

    def process_model(self, question):
        current_date_time = datetime.now()
        # inital_prompt = f"Previous conversation: {self.memory.buffer_as_str}" + f''', Thinking step by step and With only the tools provided and with the current date and time of {current_date_time} help the human with the following request, Request: {question} '''
        inital_prompt = f"""You are a researcher that uses frappe ERPnext Api to find information. Only consider Previous response if it is relevent, Ignore previous errors and failures, Thinking step by step, With only the tools provided (If a tool isnt available, use the FORWARD tool) and with the current date and time of {current_date_time} help the human with the following request, Request: {question}"""
        response = self.agent_executor.run(
            input=inital_prompt, callbacks=[self.handler]
        )

        common.bot_logging.bot_logger.info(response)

    def heartbeat(self):
        from_bot_to_bot_manager("heartbeat", os.getpid())

    def load_tools(self, llm) -> list():
        # tools = load_tools(
        #     ["human"], input_func=get_input, prompt_func=send_prompt, llm=llm
        # )
        # tools = load_tools()
        tools = []

        tools.append(ERPGETLIST())
        tools.append(ERPGETDOC())
        tools.append(ERPGETFIELDS())
        tools.append(Forward())

        return tools

    def get_credential(self, name):
        common.bot_logging.bot_logger.info(f"Fetching credential for: {name}")
        for credential in self.credentials:
            if name in credential:
                common.bot_logging.bot_logger.debug(credential[name])
                return credential[name]
        return False
