from typing import List
from dotenv import load_dotenv
from chains.AgentTypeChain import AgentTypeChain
from chains.RagPromptOptimizerChain import RagPromptOptimizerChain
from embeddings_store import Embeddings
from vectorstorage.vectorstore import get_vectorstore_for_db

from multiagent.json_extractor import extract_json, get_error_json
from typing import Any, Dict, List, Optional
import json

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun
)

from langchain.vectorstores.faiss import FAISS

load_dotenv()

class VideoTranscriptQueryConversationChain(Chain):
    llm:BaseLanguageModel
    url: str
    debug: bool
    k: int = 15
    embeddings: Embeddings = Embeddings.HUGGINGFACE
    conversation: ConversationChain = None
    conversation_sum: ConversationSummaryMemory = None

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    @property
    def input_keys(self) -> List[str]:
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        return ["output"]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, str]:
        
        if self.conversation_sum == None:

            self.conversation_sum = ConversationSummaryMemory(llm=self.llm)

            self.conversation = ConversationChain(
	            llm =self.llm,
	            memory=self.conversation_sum
                )
            
            self.conversation.prompt.template = build_start_prompt()
            
            print(f"MEMORY>{self.conversation.prompt.template}")
                
        #Get query from input
        query = inputs["query"]

        if not query:
            raise ValueError("query not set")

        #Assertain prompt type - only RAG is supported for now
        process_chain = AgentTypeChain(llm=self.llm, debug=True)
        process_response = process_chain.run({"query": query})["process"]

        print(f"Process response: {process_response}")

        if process_response == "SUMMARY":
            print("SUMMARY not yet supported")
            return self.build_return("That question would require a summarisation of the whole video, which is not yet supported.", [], None)
        
        elif process_response == "NONE":
            response = self.conversation(query)
            response["response"].replace("\r", "")
            return self.build_return(response, [], query)

        #Optimize RAG search prompt
        optimizer_chain = RagPromptOptimizerChain(llm=self.llm, debug=True)
        optimizer_response_json = optimizer_chain.run({"query": query})

        optimizer_response = optimizer_response_json["searchprompt"]

        #Get docs from db
        db = get_vectorstore_for_db(self.url, Embeddings.HUGGINGFACE)

        docs = db.similarity_search(optimizer_response, k=self.k)
        docs_page_content = [doc.page_content for doc in docs]

        #Send prompt to LLM
        prompt = build_prompt(query, docs_page_content)
        
        response = self.conversation(prompt)
        response = response["response"].replace("\r", "")        

        print(f"BEFORE {self.conversation.prompt}")

        return self.build_return(response, docs, self.conversation.memory.prompt)
        
    def build_return(self, response, docs, prompt):
        return {
            "output":{
                "response": response,
                "metadata": {
                    "docs": docs,
                    "prompt": prompt
                }
            }
        }

def build_prompt(query, docs):

    template = PromptTemplate(
        input_variables=["query", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos
        based on the video's transcript.

        Answer the following user's question:
        "{query}"

        Answer by searching the following video transcript very carefully, ensuring to be as helpful and comprehensive as possible :
        Transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, only answer "I don't know", and don't rabbit on about it.

        Your answers should be verbose and detailed, unless you are answering "I don't know", in which case you should be brief.

        Complete your sentence, don't leave it hanging.

        Answer in a friendly manner, that is easy to understand.        
        """)

    message = template.format(query=query, docs=docs)

    print(f"MESSAGE>>> {type(message)}")

    return message

def build_start_prompt():
    return """You are a helpful assistant that that can answer questions about youtube videos based on the video's transcript.

        Answer by searching the video transcripts very carefully, ensuring to be as helpful and comprehensive as possible
        
        RULES:

        Only use the factual information from the transcript and chat history to answer the question.

        If you feel like you don't have enough information to answer the question, only answer "I don't know", and don't rabbit on about it.

        Your answers should be verbose and detailed, unless you are answering "I don't know", in which case you should be brief.

        Complete your sentence, don't leave it hanging.

        Answer in a friendly manner, that is easy to understand.
                                                  
        Current conversation:
        {history}
        Human: {input}
        AI:"""