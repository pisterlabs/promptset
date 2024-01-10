import os
from typing import Tuple
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader #,PyPDFium2Loader commenting for future
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from integrations import llm, embeddings
from prompts import rag_multilingual_prompt

class TextQuery:
    """
    TextQuery allows querying and retrieving information from a collection of documents.

    It provides methods to ingest a document, ask questions, and retrieve relevant answers based on the ingested documents.
    """

    def __init__(self) -> None:
        """
        Initialize the TextQuery class.
        """
        self.embeddings = embeddings
        # I prefer MarkdownHeaderTextSplitter over RecursiveCharacterTextSplitter
        # self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50,separators=["#", "##", "###","\n\n", "\n", "(?<=\. )", " ", ""])
        self.text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ])
        self.llm = llm
        self.rag = None
        self.db = None
        # Can be saved for restarting
        #self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.history = []

    def ask(self, question: str) -> Tuple[str,str]:
        """
        Ask a question and get a response along with relevant context.

        Args:
            question (str): The question to ask.

        Returns:
            str: The response to the question.
        """
        if self.rag is None:
            response = "Please, add a document."
        else:
            result = self.rag({"question": question, "chat_history": self.history})
            response = result["answer"]
            self.history.append((question, response))
            source = result['source_documents'][0].page_content

        return response, source

    def ingest(self, file_path: os.PathLike) -> None:
        """
        Ingest a document file and prepare it for retrieval.

        Args:
            file_path (os.PathLike): The path to the document file.
        """
        db_dir = f"./{file_path}_db"
        if not os.path.isdir(db_dir):
            loader = TextLoader(file_path)
            documents = loader.load()
            splitted_documents = self.text_splitter.split_text(documents[0].page_content)
            self.db = Chroma.from_documents(splitted_documents, self.embeddings, persist_directory=db_dir).as_retriever()
        else:
            self.db = Chroma(persist_directory=db_dir, embedding_function=self.embeddings).as_retriever()
        self.rag = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.db,
            #self.memory,
            combine_docs_chain_kwargs={'prompt': rag_multilingual_prompt},
            return_source_documents=True
        )

    def forget(self) -> None:
        """
        Forget the current document and reset the retriever.
        """
        self.db = None
        self.rag = None
        #self.memory = None
        self.history = None
