import os
import openai

from pathlib import Path
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from src.utils import *

load_dotenv()
openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")


class Agent:
    def __init__(
        self, name: str, model: str, temperature: float, parser: dict, prompt: Path
    ) -> None:
        self.__name: str = name

        self.character: str = self.__load_agent_prompt(prompt)
        self.chain: LLMChain = self.__setup_chain(model, temperature)
        self.parser: dict = parser

    @property
    def name(self):
        return self.__name

    def __load_agent_prompt(self, prompt: Path) -> str:
        with open(prompt, "r") as f:
            agent_prompt = f.read()

        return agent_prompt

    def __setup_chain(self, model: str, temperature: float) -> LLMChain:
        llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
        )

        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.character),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{message}"),
            ]
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        return LLMChain(llm=llm, prompt=prompt, verbose=False, memory=memory)

    def inject_message(self, text: str, kind: str = "human") -> None:
        if kind == "human":
            message = HumanMessage(content=text)
        elif kind == "ai":
            message = AIMessage(content=text)
        elif kind == "system":
            message = SystemMessage(content=text)
        else:
            raise ValueError(
                "This message type is not supported. Use one of ['human', 'ai', 'system']"
            )

        self.chain.memory.chat_memory.add_message(message)

    def answer(self, message: str, verbose=False):
        answer = self.chain.run({"message": message})

        if verbose:
            print(f"\033[34m{self.name}:\033[0m {answer}")

        return answer


class ConversationWrapper:
    """
    Implementation of the conversation between two agents. agent1 must be the agent that returns the final results
    """

    def __init__(self, agent1: Agent, agent2: Agent, max_turns: int = 5) -> None:
        self.agent1: Agent = agent1
        self.agent2: Agent = agent2
        self.max_turns: int = max_turns

    def start(self, user_query: str):
        current_response = user_query
        for _ in range(self.max_turns):
            agent1_response = self.agent1.answer(current_response, verbose=True)
            agent1_response = parse_response(agent1_response, self.agent1.parser)
            if type(agent1_response) == dict and agent1_response["accepted"] == True:
                return agent1_response
            elif type(agent1_response) == dict and agent1_response["accepted"] == False:
                agent1_response = agent1_response["text"]
            # if agent1_response.strip().startswith(
            #     "__END__"
            # ):  # either it ends directly (like for documentation)...
            #     return agent1_response

            current_response = self.agent2.answer(agent1_response, verbose=True)
            current_response = parse_response(current_response, self.agent2.parser)
            if type(current_response) == dict and current_response["accepted"] == True:
                return agent1_response
            elif (
                type(current_response) == dict and current_response["accepted"] == False
            ):
                current_response = current_response["text"]
            # if current_response.strip().startswith(
            #     "__END__"
            # ):  # ... or someone has to approve it first (like for testing)
            #     return agent1_response


class HumanConversationWrapper:
    def __init__(self, agent1: Agent, max_turns: int = 10) -> None:
        self.agent1: Agent = agent1
        self.max_turns: int = max_turns

    def start(self):
        human_response = input("\033[34mWhat project can we develop for you?\033[0m ")
        for _ in range(self.max_turns):
            ai_response_txt = self.agent1.answer(human_response, verbose=True)
            ai_response = parse_response(ai_response_txt, self.agent1.parser)
            if type(ai_response) == dict and ai_response["accepted"]==True:
                # print("Summary: "+ ai_response["text"])
                print(
                    f"\033[34m{self.agent1.name}:\033[0m Thank you for specifying your requirements. We will start working on your project now. Stay tuned!"
                )
                return 
            elif type(ai_response) == dict and ai_response["accepted"]==False:
                pass

            human_response = input("\033[34mYour answer:\033[0m ")

            if human_response.lower() == "y":
                return
