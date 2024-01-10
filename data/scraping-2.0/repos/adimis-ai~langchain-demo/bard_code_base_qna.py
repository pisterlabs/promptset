from typing import List
from bard_llm import BardLLM
from langchain.vectorstores import Chroma
from langchain.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.document import Document
from code_splitter import CodeSplitter

class BardCodeBaseQna:
    def __init__(self, token: str, directory_path: str):
        """
        Initialize a BardCodeBaseQna instance.

        Args:
            token (str): The token for authentication with BardLLM.
            directory_path (str): The path to the directory containing code documents.
        """
        self.token = token
        self.llm = BardLLM(token=token, verbose=True)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.docs: List[Document] = self._load_documents(directory_path)
        self.retriever = self._create_retriever()

    def _load_documents(self, directory_path: str) -> List[Document]:
        """
        Load code documents from a directory.

        Args:
            directory_path (str): The path to the directory containing code documents.

        Returns:
            List[Document]: A list of Document objects representing code segments from the directory.
        """
        code_splitter = CodeSplitter(
            chunk_size=2500, 
            chunk_overlap=200, 
        )
        code_documents = code_splitter(directory_path)
        return code_documents

    def _create_retriever(self):
        """
        Create a retriever for code documents.

        Returns:
            retriever: A retriever for code documents.
        """
        try:
            embedder = SpacyEmbeddings()
            print(f"Creating retriever with {len(self.docs)} docs")
            db = Chroma.from_documents(self.docs, embedder)
            return db.as_retriever(search_type="mmr", search_kwargs={"k": 6})
        except Exception as e:
            print(f"Error creating retriever: {e}")
            raise

    def _run(self, query: str):
        """
        Run the question-answering process.

        Args:
            query (str): The question/query to answer.

        Returns:
            str: The response to the query.
        """
        qa = ConversationalRetrievalChain.from_llm(
            self.llm, self.retriever, memory=self.memory, verbose=True
        )
        prompt = (
            f"{query}.\nYou should use the codebases and contextual information provided above "
            "to formulate your response. If you do not possess the necessary information to answer the question, "
            "please be honest and state that you do not know, rather than attempting to invent an answer."
        )
        res = qa.run(prompt)
        return res

    def __call__(self, query: str):
        """
        Make the BardCodeBaseQna instance callable with a query.

        Args:
            query (str): The question/query to answer.

        Returns:
            str: The response to the query.
        """
        response = self._run(query)
        return response
