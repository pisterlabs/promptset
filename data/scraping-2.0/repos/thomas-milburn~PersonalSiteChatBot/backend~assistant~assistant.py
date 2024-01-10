from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from langchain.schema import SystemMessage

import config
from tools.contact.contact_tool import ContactTool
from tools.github.get_activity import GithubGetUserEvents
from tools.github.get_read_me import GithubGetRepoReadMe
from tools.university.university_modules import GetUniversityModules, GetUniversityModuleDescription

with open("context/system_prompt.txt", "r") as f:
    system_prompt = f.read().strip()


class AssistantChainManager:
    def __init__(self, stream_handler):
        self.stream_manager = AsyncCallbackManager([stream_handler])

        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            verbose=True,
            openai_api_key=config.config["PERSONAL_SITE_OPENAI_KEY"],
            temperature=1,
            streaming=True,
            callback_manager=self.stream_manager,
        )

        self.tools = [
            GetUniversityModules(callback_manager=self.stream_manager),
            GetUniversityModuleDescription(callback_manager=self.stream_manager),
            ContactTool(callback_manager=self.stream_manager),
            GithubGetUserEvents(callback_manager=self.stream_manager),
            GithubGetRepoReadMe(callback_manager=self.stream_manager)
        ]

        self.memory = ConversationTokenBufferMemory(llm=self.llm,
                                                    memory_key="chat_history",
                                                    max_token_limit=1024,
                                                    return_messages=True)

    def get_chain(self) -> AgentExecutor:
        # Step 3: Create the prompt
        system_message = SystemMessage(content=system_prompt)
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=self.memory.buffer_as_messages
        )

        # Step 4: Create the agent
        agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=prompt)

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            callback_manager=self.stream_manager
        )

        return agent_executor
