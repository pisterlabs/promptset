from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from typing import Union, List
from apikey import load_env

OPENAI_API_KEY, SERPER_API_KEY = load_env()

class ChatService:
    def __init__(
            self,
            llm : Union[ChatOpenAI, OpenAI] = None,
            temperature : float = 0
    ):
        if llm is None:
            llm = OpenAI(model='gpt-3.5-turbo', temperature=temperature, openai_api_key=OPENAI_API_KEY)
        self.llm = llm

    def chat(
            self,
            prompt,
    ) -> str:
        response = self.llm(messages=prompt)
        return response.content

    def chat_with_template(
            self,
            template : List[dict]
    ) -> str:
        message = []
        for command in template:
            message.append(self.__construct_message__(command))
        
        return self.llm(message)
    
    def __construct_message__(
            self,
            command: dict
    ):
        if command['role'] == 'system':
            return SystemMessage(
                content= command['content']
            )
        if command['role'] == 'human':
            return HumanMessage(
                content=command['content']
            )
        if command['role'] == 'AI':
            return AIMessage(
                content=command['content']
            )