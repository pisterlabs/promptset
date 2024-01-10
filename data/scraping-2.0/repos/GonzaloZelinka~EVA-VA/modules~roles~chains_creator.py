from abc import ABC, abstractmethod
from .chains import ChainGeneral, MeetingChain, Q_AChain
from typing import Dict
from modules.roles_templates.improve_listen_template import (
    human_improve_listening_template,
    system_improve_listening_template,
)
from modules.openai_functions.request_improve import improved_req_fn
from modules.functions.create_prompt import create_prompt
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from dotenv import load_dotenv

load_dotenv()


class ChainCreator(ABC):
    # def __init__(self):
    #     self.__chain = None
    #     self.__llm = ChatOpenAI(temperature=0.1)

    def execute(self, improved_req) -> str:
        chain_type = self._identify_subtask(improved_req)
        self.__chain = self._factory_chain(chain_type)
        return self.__chain.execute_chain(improved_req)

    @abstractmethod
    def _factory_chain(self):
        pass

    @abstractmethod
    def _identify_subtask(self, improved_req):
        pass


class FactoryChain:
    def __init__(self):
        self.__creators = {}
        self.__llm = ChatOpenAI(temperature=0.3)

    def reg_concrete_chain(self, request_type: str, creator: ChainCreator):
        self.__creators[request_type] = creator

    def create_concrete_chain_creator(
        self, req_text: str
    ) -> tuple[ChainCreator, Dict[str, str]]:
        prompt = create_prompt(
            system_prompt=system_improve_listening_template,
            human_prompt=human_improve_listening_template,
            input_variables=["output"],
        )
        functions_chain = (
            prompt
            | self.__llm.bind(
                function_call={"name": "req_improved"}, functions=[improved_req_fn]
            )
            | JsonOutputFunctionsParser()
        )
        requestEnhanced = functions_chain.invoke({"output": req_text})
        creator = self.__creators.get(requestEnhanced["req_type"])
        if not creator:
            raise ValueError(requestEnhanced)
        return creator, requestEnhanced


class ChainMeetingCreator(ChainCreator):
    def _identify_subtask(self, improved_req):
        return "meeting"

    def _factory_chain(self, chain_type) -> ChainGeneral:
        if chain_type == "meeting":
            return MeetingChain()
        else:
            raise ValueError("Unknown chain type")


class ChainQ_ACreator(ChainCreator):
    def _identify_subtask(self, improved_req):
        return "Q_A"

    def _factory_chain(self, chain_type) -> ChainGeneral:
        return Q_AChain()
