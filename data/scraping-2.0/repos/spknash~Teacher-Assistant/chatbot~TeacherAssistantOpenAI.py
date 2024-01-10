

import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema.chat_history import*

OPENAI_KEY = os.environ.get('OPENAI_KEY')

class TeacherAssistantOpenAI:

    def __init__(self, message_list):

        OPENAI_MODEL = "gpt-4"

        TEMPLATE = """
        I want you to act as an Computer Sience Teacher Assitant. I will now provide you
        with some code I want you to analyse and understand.   
        {completed_code}

        A student is currently trying to replicate the code above. He was provided this boiler plate:
        {boilerplate}

        The student is currently stuck and has asked you for help. He has provided the following question:
        {question}

        Answer the student's question by giving a brief response and not giving away too much information on how to complete the task.
        """

        prompt = PromptTemplate(
            input_variables=["completed_code", "boilerplate", "question"],
            template=TEMPLATE

        )

        llm = OpenAI(model_name=OPENAI_MODEL, temperature=0.9, openai_api_key=OPENAI_KEY)
        print("before instantiat")
    
        print("after")
        self.memory = ConversationBufferMemory(input_key = 'question', memory_key='chat_history', max_len=1000)
        index = 0
        # Split message_list into user_message_list ai_message_list
        user_message_list = []
        ai_message_list = []

        for message in message_list:
            if index%2 == 0:
                user_message_list.append(message)
            else:
                ai_message_list.append(message)
            index += 1
        print(len(user_message_list))
        print(len(ai_message_list))
        for i in range(len(user_message_list)):
            self.memory.save_context({"question":user_message_list[i]},{"chat_history":ai_message_list[i]})


        self.llm_chain = LLMChain(llm=llm, verbose=True,memory=self.memory, prompt=prompt)
        
# class FileChatMessageHistory(BaseChatMessageHistory):
        
#     def add_ai_message(self, message: str) -> None:
#         return super().add_ai_message(message)
        
#     def add_user_message(self, message: str) -> None:
#         return super().add_user_message(message)
        

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


    
    
