
"""
ChatBot classes
"""

# import random
from openai import OpenAI
# import langchain.llms 
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from info_files.meds import meds_data
from util import local_settings
from info_files.prompt_list import template

# OpenAI API ------------------------------------------------------------------------------------------------------------------------------------
class GPT_Helper:
    def __init__(self, OPENAI_API_KEY: str, system_behavior: str="", functions: list=None, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.messages = []
        self.model = model
        self.system_behavior = system_behavior
      
        if system_behavior:
            self.messages.append({"role": "system", "content": system_behavior})

        if functions:
            self.functions = functions

        self.document_searcher = FAISS.from_texts(meds_data, OpenAIEmbeddings())
        self.condense_question_prompt_template = PromptTemplate.from_template(template)
        self.qa_prompt = PromptTemplate(template=self.system_behavior, input_variables=["context", "question"])
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.llm = ChatOpenAI(temperature=0.1)


    # get completion from the model
    def get_completion(self, prompt, temperature=0, is_langchain=False):

        self.messages.append({"role": "user", "content": prompt})

        # Get completion from the model
        if not is_langchain:
            completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=temperature,
            tools=self.functions)

            if completion.choices[0].message.content:
                self.messages.append({"role": "assistant", "content": completion.choices[0].message.content})

            return completion.choices[0].message

        # Get completion from the langchain
        else:
            question_generator = LLMChain(llm=self.llm, prompt=self.condense_question_prompt_template, memory=self.memory)

            doc_chain = load_qa_chain(self.llm, chain_type="stuff", prompt=self.qa_prompt)

            qa_chain = ConversationalRetrievalChain(
                retriever= self.document_searcher.as_retriever(search_kwargs={'k': 6}),
                question_generator=question_generator,
                combine_docs_chain=doc_chain,
                memory=self.memory,
                )

            result = qa_chain({'question': prompt, 'chat_history': self.messages})
            response = result['answer']

            self.messages.append({"role": "assistant", "content": response})

            return response



# ChatBot ---------------------------------------------------------------------------------------------------------------------------------------

class DrChatBot:
    """
    Generate a response by using LLMs.
    """

    def __init__(self, system_behavior: str, functions: list=None):
        self.__system_behavior = system_behavior
        self.__functions = functions

        self.engine = GPT_Helper(
            OPENAI_API_KEY=local_settings.OPENAI_API_KEY,
            system_behavior=system_behavior,
            functions=functions
        )

    def generate_response(self, message: str, is_langchain=False):
        return self.engine.get_completion(message, is_langchain = is_langchain)

    def __str__(self):
        shift = "   "
        class_name = str(type(self)).split('.')[-1].replace("'>", "")

        return f"🥼🤖 {class_name}."

    def reset(self):
        self.engine = GPT_Helper(
            OPENAI_API_KEY=local_settings.OPENAI_API_KEY,
            system_behavior=self.__system_behavior,
            functions=self.__functions
    )

    @property
    def memory(self):
        return self.engine.messages

    @property
    def system_behavior(self):
        return self.__system_config

    @system_behavior.setter
    def system_behavior(self, system_config : str):
        self.__system_behavior = system_config




