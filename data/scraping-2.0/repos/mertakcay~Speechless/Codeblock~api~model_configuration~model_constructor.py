from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage

from os.path import expanduser

from langchain.llms import LlamaCpp



class modelConstructor():
    def __init__(self, system_message = "You are a machine learning enginner. You have to help me to improve my prompt templates.",
                model_path = "./model_configuration/model/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                is_streaming = False):

        
        template_messages = [
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
        
        prompt_template = ChatPromptTemplate.from_messages(template_messages)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        llm = LlamaCpp(
            model_path=model_path,
            streaming=False,
        )
        
        
        model = Llama2Chat(llm=llm)
        self.chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)
        
    def get_chain(self):
        return self.chain