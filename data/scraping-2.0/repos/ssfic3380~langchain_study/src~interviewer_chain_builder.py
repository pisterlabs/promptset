from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from handler.custom_streaming_handler import CustomStreamingCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

from dotenv import load_dotenv
load_dotenv('../config.env')


class InterviewerChainBuilder:
    def __init__(self, company_name, job_name, job_posting, cover_letter):
        self.company_name = company_name
        self.job_name = job_name
        self.job_posting = job_posting
        self.cover_letter = cover_letter

        self.chat = ChatOpenAI(temperature=0.8,
                               streaming=True,
                               callbacks=[CustomStreamingCallbackHandler()],
                               verbose=True)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def create_system_message_prompt(self):
        info_template = f"""
        당신은 {self.company_name}라는 회사에서 {self.job_name} 포지션의 신입 사원을 뽑는 면접관의 역할을 수행하게 됩니다. 아래는 이번 모집의 세부 사항과, 회사에서 제시한 질문에 대해서 대답한 지원자의 자기소개서입니다. 이 정보들을 주의 깊게 분석하고 이해하며, 면접관으로서의 역할을 준비하십시오.

        ```
        회사명: {self.company_name}
        직무명: {self.job_name}

        모집 공고:
        {self.job_posting}
        
        지원자의 자기소개서:
        {self.cover_letter}
        ```
        """
        return SystemMessagePromptTemplate.from_template(info_template)

    def create_human_message_prompt(self):
        user_input_template = """
        {user_input}
        """
        return HumanMessagePromptTemplate.from_template(user_input_template)

    def create_chat_prompt(self):
        return ChatPromptTemplate.from_messages(
            [self.create_system_message_prompt(), self.create_human_message_prompt()])

    def get_full_chain(self):
        return LLMChain(llm=self.chat,
                        prompt=self.create_chat_prompt(),
                        verbose=True,
                        memory=self.memory)