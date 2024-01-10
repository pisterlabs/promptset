import openai
import asyncio
import datetime

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.schema import AIMessage, HumanMessage

from src.ai.data.params import prompt as prompt_text, model, temperature
from src.ai.tools import search_from_documents
from src.ai.utils import format_text_to_html

from src.config import OPENAI_API_KEY, DEBUG


class BaseAIChatBot:
    """Base class for LLM-powered chat bots"""

    def __init__(
        self,
        openai_api_key: str,
        model: str,
        temperature: float,
        tools: list = [],
        prompt: str = None,
    ) -> None:
        self.openai_api_key = openai_api_key
        self.tools = tools
        self.model = model
        self.temperature = temperature
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        self.memory_with_date = []

    async def create_agent(self) -> AgentExecutor:
        """
        Create agent to execute LLM operations
        """
        llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model=self.model,
            temperature=self.temperature,
        )
        llm_with_tools = llm.bind(
            functions=[format_tool_to_openai_function(t) for t in self.tools]
        )
        agent_schema = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_functions(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | self.prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        return AgentExecutor(
            agent=agent_schema,
            tools=self.tools,
            memory=self.memory,
            verbose=DEBUG,
            handle_parsing_errors=True,
        )

    async def get_history(self):
        return self.memory_with_date

    async def set_history(self, history):
        self.memory.buffer.extend(self.convert_json_to_memory(history))
        self.memory_with_date = history

    @staticmethod
    def convert_conversation_to_json(history):
        serialized = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                serialized.append(
                    {
                        "role": "HumanMessage",
                        "content": msg.content,
                        "date": datetime.datetime.now().isoformat(),
                    }
                )
            elif isinstance(msg, AIMessage):
                serialized.append(
                    {
                        "role": "AIMessage",
                        "content": msg.content,
                        "date": datetime.datetime.now().isoformat(),
                    }
                )
        return serialized

    @staticmethod
    def convert_json_to_memory(data):
        serialized = []
        for msg in data:
            if msg["role"] == "HumanMessage":
                serialized.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "AIMessage":
                serialized.append(AIMessage(content=msg["content"]))
        return serialized

    async def agent_answer_loop(self, message: str) -> str:
        agent = await self.create_agent()

        self.memory_with_date.append(
            {
                "role": "HumanMessage",
                "content": message,
                "date": datetime.datetime.now().isoformat(),
            }
        )

        try:
            response = agent.invoke({"input": message})
        except asyncio.TimeoutError:
            return {
                "message": "I took too long to think, let's try again.",
                "status": 503,
            }
        except openai.error.APIError:
            return {
                "message": "I'm overloaded right now, please try again later.",
                "status": 507,
            }
        except openai.error.InvalidRequestError:
            return {
                "message": "My memory is overloaded, please start a new conversation.",
                "status": 509,
            }

        response = format_text_to_html(response["output"])

        self.memory_with_date.append(
            {
                "role": "AIMessage",
                "content": response,
                "date": datetime.datetime.now().isoformat(),
            }
        )

        return {"message": response, "status": 200}

    async def send_message(self, message: str) -> str:
        """
        Send bot answer as text response
        """
        response = await self.agent_answer_loop(message=message)
        return response


class MarketingAIBot(BaseAIChatBot):
    def __init__(self):
        super().__init__(
            OPENAI_API_KEY, model, temperature, [search_from_documents], prompt_text
        )
