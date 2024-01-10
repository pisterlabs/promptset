

import os
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema.chat_history import*
import streamlit as st




MODEL_ID = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT = """
    You are a very helpful Computer Sience Teacher Assitant with the goal of helping a student recreate a coding project. 
    The student has been provided with some boiler plate code is currently trying to replicate the completed project.
    You will use the provided context of both the given boiler plate code and the completed code to help the student.
    Before answering the student question read both the boiler plate code and the completed code to understand the whole code base. 
    When answering the question you do not need to refer to code in the boiler plate code or completed code.
    Make sure you never give the context of the completed code to the student.
    When replying to the student make sure you are talking to him like you are a teacher assistant helping a student.
    """


SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


INSTRUCTION_PROMPT = """

    Boilerplate Code: {boilerplate}\n
    Completed Code: {completed_code}\n
    Chat History: {chat_history}\n
    Student Question: {question}\n
    """

TEMPLATE = B_INST + SYSTEM_PROMPT + INSTRUCTION_PROMPT + E_INST

class TeacherAssistantLlama2:

    def __init__(self, message_list):

        
        prompt = PromptTemplate(
            input_variables=["completed_code", "boilerplate", "question"],
            template=TEMPLATE

        )

        llm = Replicate(model=MODEL_ID,
        model_kwargs={"temperature": 0.75, "max_length": 5000, "top_p": 1})
        
        self.memory = ConversationBufferMemory(input_key = 'question', memory_key='chat_history', max_len=1000)
    

        # Split message_list into user_message_list ai_message_list
        user_message_list = []
        ai_message_list = []

        for index, message in enumerate(message_list):
            if index%2 == 0:
                user_message_list.append(message)
            else:
                ai_message_list.append(message)
    
    
        for i in range(len(user_message_list)):
            self.memory.save_context({"question":user_message_list[i]},{"chat_history":ai_message_list[i]})


        self.llm_chain = LLMChain(llm=llm, verbose=True, memory=self.memory, prompt=prompt)
        

    def query_chain(self, question, completed_code, boilerplate):
        return self.llm_chain.run(question=question, completed_code=completed_code, boilerplate=boilerplate)
    
    def get_memory(self):
        message_list_str = []
        message_list =  self.memory.buffer_as_messages

        for message in message_list:
            message_list_str.append(message.content)

        return message_list_str

    
    def reset_memory(self):
        self.memory.clear()



# ta = TeacherAssistant([])
# print("OK")
# ta.query_chain("What should go in print?", "print('hello')","print()")
# ta.query_chain("What function should I use", "print('hello')","print()")
# message_list = ta.get_memory()

# print(message_list)





