"""
Gemini DM

Gemini AI based dungeon master
"""
import logging

logging.basicConfig(format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s")

from pathlib import Path
from concurrent.futures import as_completed

import faiss

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain

from player import Player


class GeminiDM:
    def __init__(self, player: any = None):
        self.instruction_prompt_path = Path("prompts/dmstart.txt")
        self.story_path = Path("data/story.txt")
        self.player = player if player else Player()
        self.llm = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-pro")
        self.conversation = None
        self.chain_recorder = None

        # setup logging
        self.class_logger = logging.getLogger(__name__)
        self.class_logger.setLevel(logging.DEBUG)

        # get story
        story_txt = ""
        with open(self.story_path) as txtfile:
            for tline in txtfile.readlines():
                story_txt += tline

        # setup instruction prompt
        prompt_txt = ""
        with open(self.instruction_prompt_path) as txtfile:
            for tline in txtfile.readlines():
                prompt_txt += tline

        # build prompt with player information and 
        # for the chat buffer
        prompt_txt += f"""
        Player Info:
        {self.player.player_sheet()}\n"""

        prompt_txt += """
        Current Conversation:
        {history}
        
        Human: {input}
        AI:"""

        self.instruction_prompt_template = PromptTemplate(
            input_variables=["history", "input"], template=prompt_txt
        )

        # setup chat vectorstore
        # this is using faiss-cpu
        embedding_size = 768
        self.vectorstore = FAISS(
            VertexAIEmbeddings().embed_query,
            faiss.IndexFlatL2(embedding_size),
            InMemoryDocstore({}),
            {},
        )

        # setup memory
        retriever = self.vectorstore.as_retriever(search_kwargs=dict(k=3))
        self.memory = VectorStoreRetrieverMemory(
            retriever=retriever
        )

        # add blank context to kickstart things
        self.memory.save_context({"input": ""},{"output": ""})

        # creating llm chain
        self.conversation = ConversationChain(
            llm=self.llm,
            prompt=self.instruction_prompt_template,
            memory=self.memory,
            verbose=True,
        )
    def chat(self, user_msg: str) -> str:
        """
        String input to gemini chat from user
        Record chat interaction and test for conciseness with TruLens
        """
        resp = self.conversation.invoke(user_msg)
        return resp["response"]
