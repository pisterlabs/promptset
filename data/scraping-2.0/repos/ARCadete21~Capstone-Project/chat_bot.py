"""
ChatBot classes
"""

import random
from openai import OpenAI
from util import local_settings
from GetAllData import vector_store
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from ChatbotFunctions import tools_custom

# [i]                                                                                            #
# [i] OpenAI API                                                                                 #
# [i]                                                                                            #

####################################
class GPT_Helper:

    def __init__(self,
                 OPENAI_API_KEY: str,
                 system_behavior: str = "",
                 model="gpt-3.5-turbo",
                 tools: list = None,
                 ):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.messages = []
        self.model = model
        self.tools = tools
        self.system_behavior = system_behavior

        self.vectorStore = vector_store
        self.BufferMemory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.llm = ChatOpenAI(temperature=0.2)#, tools=self.tools)

        if system_behavior:
            self.messages.append({
                "role": "system",
                "content": system_behavior
            })

    # [i] get completion from the model             #
    def get_completion_old(self, prompt, temperature=0):
        self.messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=temperature,
            tools=self.tools
        )

        self.messages.append(
            {
                "role": "assistant",
                "content": completion.choices[0].message.content
            }
        )

        return completion.choices[0].message.content

    def get_completion(self, prompt):
        self.messages.append({"role": "user", "content": prompt})

        crc = ConversationalRetrievalChain.from_llm(self.llm, retriever=self.vectorStore.vectorstore.as_retriever(),
                                                    memory=self.BufferMemory )

        prompt_result = crc({'question': prompt, 'chat_history': self.messages})
        answer = prompt_result['answer']

        self.messages.append({"role": "assistant", "content": answer})

        return answer


# [i]                                                                                            #
# [i] F1 Chat Bot                                                                                #
# [i]                                                                                            #

class F1ChatBot:
    """
    Generate a response by using LLMs.
    """

    def __init__(self, system_behavior: str):
        self.__system_behavior = system_behavior

        self.engine = GPT_Helper(
            OPENAI_API_KEY=local_settings.OPENAI_API_KEY,
            system_behavior=system_behavior,
            tools= tools_custom
        )

    def generate_response(self, message: str):
        return self.engine.get_completion(message)

    def __str__(self):
        shift = "   "
        class_name = str(type(self)).split('.')[-1].replace("'>", "")

        return f"🤖🏎️ {class_name}."

    def reset(self):
        self.engine = GPT_Helper(
            OPENAI_API_KEY=local_settings.OPENAI_API_KEY,
            system_behavior=self.system_behavior,
            tools=tools_custom
        )

    @property
    def memory(self):
        return self.engine.messages

    @property
    def system_behavior(self):
        return self.__system_config

    @system_behavior.setter
    def system_behavior(self, system_config: str):
        self.__system_behavior = system_config
