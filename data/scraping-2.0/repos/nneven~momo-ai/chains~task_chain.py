import config
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

class TaskChain:
    def __init__(self, syllabus: str, llm: ChatOpenAI = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)):
        self.syllabus = syllabus
        self.llm = llm
        human_message_prompt = HumanMessagePromptTemplate(
            prompt = PromptTemplate(
                input_variables=["syllabus", "question"],
                template="Given the following syllabus information: {syllabus}, please answer the following question about the class: {question}"
            )
        )
        self.prompt = ChatPromptTemplate.from_messages([human_message_prompt])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def run(self, question: str):
        return self.chain.run({"syllabus": self.syllabus, "question": question})
