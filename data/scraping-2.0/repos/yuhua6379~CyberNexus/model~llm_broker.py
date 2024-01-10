import os

import openai

from common.base_thread import get_logger
from datasource.config import rdbms_instance
from datasource.rdbms.entities import ChatLogModel
from model.llm import BaseLLM
from model.llm_session import LLMSession
from repo.character import Character


class LLMBroker:

    def __init__(self, llm: BaseLLM,
                 character1: Character,
                 character2: Character,
                 version: str = '0.0.1'):
        self.llm = llm
        self.character1 = character1
        self.character2 = character2
        self.version = version

    def chat(self, session: LLMSession) -> LLMSession:
        prompt = session.prompt
        message_in = self.on_chat(prompt)
        openai.api_key = os.environ['openai_api_key']

        try_times = 1
        while True:
            message_out = self.llm.chat(message_in)
            session.set_result(message_out)
            try:
                context = session.get_context()
                if context.status != 0:
                    # 如果callback函数返回了非0，终止正常流程
                    break
                else:
                    session.get_context().valid()

                # 只要格式正确，立马中断
                break
            except Exception as e:
                # 出现格式解析异常了
                if try_times >= 3:
                    raise e
                else:
                    get_logger().debug(
                        f"llm return incorrect json format retry: {try_times} message_out: {message_out}")

            try_times += 1

        # get_logger().debug(f"[[final message]]: {final_message}")
        self.after_chat(message_in, message_out)

        return session

    def on_chat(self, input_: str) -> str:
        return input_

    def after_chat(self, message_in: str, message_out: str):
        log = ChatLogModel()
        log.character1_id = self.character1.id
        log.character2_id = self.character2.id
        log.character1_message = message_in
        log.character2_message = message_out

        # 记录llm_broker的版本，便于筛选数据
        log.version = self.version

        with rdbms_instance.get_session() as session:
            session.add(log)
            session.commit()


class LLMBrokerBuilder:

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def build(self, character1: Character, character2: Character) -> LLMBroker:
        # prompt = OpenAIFunctionsAgent.create_prompt(system_message=SystemMessage(content=prompt))
        # langchain_llm_broker = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=prompt)

        return LLMBroker(llm=self.llm,
                         character1=character1,
                         character2=character2)
