import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
from typing import List, Tuple
from langchain.chains.llm import LLMChain
from langchain.llms import OpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.chat_vector_db.base import ChatVectorDBChain
from simple_chat_pdf.constant import PINE_CONE_INDEX_NAME, VECTOR_QUERY_NAME_SPACE, OPENAI_API_KEY
from simple_chat_pdf.constant import PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY, GPT_MODEL_TYPE, PINE_CONE_INDEX_NAME
from simple_chat_pdf.components.chrome_data import vectorstor_chroma
from simple_chat_pdf.components.faiss_data import faiss_vector_store

class QuestionHandler:
    def __init__(self):
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
       #  self.vectorstore = self.get_vectorstore(namespace=VECTOR_QUERY_NAME_SPACE)
       # self.vectorstore = vectorstor_chroma
        self.vectorstore = faiss_vector_store
        self.prompt = self.generate_prompt()
    def get_vectorstore(self, namespace: str) -> VectorStore:
        vectorstore = Pinecone.from_existing_index(index_name=PINE_CONE_INDEX_NAME,
                                                   embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
                                                   namespace=namespace)
        return vectorstore
    
    def get_answer(self, question: str, chat_history: List[Tuple[str, str]]) -> str:
        # chat_openai = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        streaming_llm = ChatOpenAI(streaming=True,
                                   callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                                   verbose=False, 
                                   model_name=GPT_MODEL_TYPE,
                                   openai_api_key=OPENAI_API_KEY,
                                   temperature=0)

        question_generator = LLMChain(llm=OpenAI(model_name=GPT_MODEL_TYPE,
                                                 openai_api_key=OPENAI_API_KEY,
                                                 temperature=0),
                                      prompt=CONDENSE_QUESTION_PROMPT)

        doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=self.prompt)
        qa = ChatVectorDBChain(vectorstore=self.vectorstore,
                               combine_docs_chain=doc_chain, 
                               question_generator=question_generator,
                               return_source_documents=True,
                               k=2,
                               )
        result = qa({"question": question, "chat_history": chat_history})
        return result
        


    def generate_prompt(self) -> ChatPromptTemplate:

        system_prompt_msg_template = SystemMessagePromptTemplate.from_template(
            "You are an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question. "
            "Provide a conversational answer based on the context provided.\n"
            "You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks.\n"
            "If you can't find the answer in the context below, just say \"Hmm, I'm not sure.\" Don't try to make up an answer.\n"
            "If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.\n\n"
            "Question: {question}\n"
            "=========\n"
            "{context}\n"
            "=========\n"
            "Answer in Markdown:"
        )

        messages = [
            system_prompt_msg_template,
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt
