from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain

from dotenv import load_dotenv
import logging
from . import BaseStrategy

logger= logging.getLogger(__name__)

load_dotenv()

class WebSearchStrategy(BaseStrategy):
    def __init__(self):
        self.name = "web_search"
        self.model = "gpt-3.5-turbo-16k"
    
    def determine_answer(self, question: str, choices: List[str]) -> str:
        # vec store
        vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai")

        # LLM
        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=2048)
        llm = ChatOpenAI(model_name=self.model, max_tokens=2048)

        # Search 
        search = GoogleSearchAPIWrapper()

        # Retriever
        web_research_retriever = WebResearchRetriever.from_llm(
            vectorstore=vectorstore,
            llm=llm, 
            search=search)

        choices = "\n".join(choices)
        
        user_input = f"""
        {question}. Answer in a single letter from the options below.

        {choices}
        """
        logger.info(f"User Input: {user_input}")
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm,
            retriever=web_research_retriever,
            reduce_k_below_max_tokens=True
        )
        result = qa_chain({"question": user_input})
        answer = result['answer']
        answer = answer.strip("\n").strip(" ")[0].upper()
        if answer not in ["A", "B", "C", "D"]:
            raise Exception(f"Wrong Choice! Investigate ChatGPT response... (answer: {result['answer']})")
        
        return answer