import yaml
import contextlib
import io
import traceback

from typing import List, Any
from logging import getLogger
from .config import DefaultConfig
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate

from langchain.chains import RetrievalQA
from langchain.chains import ChatVectorDBChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ReadOnlySharedMemory, ConversationBufferMemory



from .vector import DjangoVectorStore
from .free_tool import ChatDjangoDBChain


logger = getLogger(__name__)

    

    
class PaperSuperClient():
    templates_path = 'llmprompt/paperchat.yaml'

    def __init__(self, config=DefaultConfig()) -> None:
        self.config = config
        try:
        # Read YAML file
            with open(self.templates_path, 'r') as file:
                data = yaml.safe_load(file)
                self.templates = data
        except FileNotFoundError:
            # Handle the case when the file is not found
            self.templates =  {}

    def get_chat_history(self, history: List[Any]):

        return " ".join(history)

        

    def send_message(self, userprofile, transaction, history: List[Any], chat_type,only: [List] = None):

        response = self._send_message(
            userprofile=userprofile,
            transaction=transaction,
            history=history, 
            only=only,
            chat_type = chat_type
            )

        return response
    


    def make_chain_dj(self,userprofile, transaction,mode="general", only=None, strategy="cosine" ):

        llm = ChatOpenAI(
            temperature=0, 
            model_name=self.config.model_name, 
            openai_api_key=self.config.api_key, 
            verbose=True)

        embedding_func = OpenAIEmbeddings(openai_api_key=self.config.api_key, model="text-embedding-ada-002")
        
        vectorstore = DjangoVectorStore(
            userprofile=userprofile,
            transaction=transaction,
            embedding_function=embedding_func,
            search_in =only,
            strategy=strategy, 
            mode=mode)

        memory = ConversationBufferMemory(memory_key="chat_history")

        condese_prompt = PromptTemplate.from_template (self.templates["condense_prompt"])

        qa_prompt = PromptTemplate.from_template(self.templates["qa_prompt"])

        question_generator = LLMChain(memory=memory,llm=llm, prompt=condese_prompt)

        do_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=qa_prompt)

        return ChatDjangoDBChain(
            vectorstore= vectorstore,
            mode=mode,
            combine_docs_chain=do_chain,
            question_generator=question_generator,
            return_source_documents=True,
            top_k_docs_for_context=4,
            userprofile=userprofile,
            transaction=transaction,
            # memory=memory
        )

    def make_chain_dj_law(self,userprofile, transaction,mode="LAW", only=None, strategy="cosine" ):

        llm = ChatOpenAI(
            temperature=0, 
            model_name=self.config.model_name, 
            openai_api_key=self.config.api_key, 
            verbose=True)

        embedding_func = OpenAIEmbeddings(openai_api_key=self.config.api_key, model="text-embedding-ada-002")
        
        vectorstore = DjangoVectorStore(
            userprofile=userprofile,
            transaction=transaction,
            embedding_function=embedding_func,
            search_in =only,
            strategy=strategy, 
            mode=mode)

        memory = ConversationBufferMemory(memory_key="chat_history")

        condese_prompt = PromptTemplate.from_template (self.templates["condense_prompt"])

        qa_prompt = PromptTemplate.from_template(self.templates["qa_prompt"])

        question_generator = LLMChain(memory=memory,llm=llm, prompt=condese_prompt)

        do_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=qa_prompt)

        return ChatDjangoDBChain(
            vectorstore= vectorstore,
            mode=mode,
            combine_docs_chain=do_chain,
            question_generator=question_generator,
            return_source_documents=True,
            top_k_docs_for_context=4,
            userprofile=userprofile,
            transaction=transaction,
            # memory=memory
        )
    def make_chain_dj_science(self,userprofile, transaction,mode="SCIENCE", only=None, strategy="cosine" ):

        llm = ChatOpenAI(
            temperature=0, 
            model_name=self.config.model_name, 
            openai_api_key=self.config.api_key, 
            verbose=True)

        embedding_func = OpenAIEmbeddings(openai_api_key=self.config.api_key, model="text-embedding-ada-002")
        
        vectorstore = DjangoVectorStore(
            userprofile=userprofile,
            transaction=transaction,
            embedding_function=embedding_func,
            search_in =only,
            strategy=strategy, 
            mode=mode)

        memory = ConversationBufferMemory(memory_key="chat_history")

        condese_prompt = PromptTemplate.from_template (self.templates["condense_prompt"])

        qa_prompt = PromptTemplate.from_template(self.templates["qa_prompt"])

        question_generator = LLMChain(memory=memory,llm=llm, prompt=condese_prompt)

        do_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=qa_prompt)

        return ChatDjangoDBChain(
            vectorstore= vectorstore,
            mode=mode,
            combine_docs_chain=do_chain,
            question_generator=question_generator,
            return_source_documents=True,
            top_k_docs_for_context=4,
            userprofile=userprofile,
            transaction=transaction,
            # memory=memory
        )      
    
    def _send_message(self,userprofile,transaction,history:List[Any], only: List[Any]=None, mode: str="general", chat_type:str = "GENERAL"):
        
        print({"history": history, "human_input": transaction.prompt })


        if chat_type == "GENERAL":


            chain = self.make_chain_dj(
                userprofile=userprofile, 
                mode=chat_type,
                transaction=transaction,
                only=only)
        elif chat_type == "SCIENCE":
            chain = self.make_chain_dj_science(
                userprofile=userprofile, 
                mode=chat_type,
                transaction=transaction,
                only=only)
            
        elif chat_type == "LAW":
            chain = self.make_chain_dj_law(
                userprofile=userprofile, 
                mode=chat_type,
                transaction=transaction,
                only=only)
            
        else:

            chain = self.make_chain_dj(
                userprofile=userprofile, 
                mode=chat_type,
                transaction=transaction,
                only=only)
        #sanitized_question = sanitize_sentence(transaction.prompt)
        response = {}
        try:
            result= chain({
            "question": transaction.prompt,
            "chat_history": history
                })
            
            print({"history": history, "human_input": transaction.prompt })

            response['message'] = result['answer']
            response['sources'] = result['source_documents']

        except Exception as e:

            response['error'] = "failed , {}".format(e)


        print(response)
        return response
        

