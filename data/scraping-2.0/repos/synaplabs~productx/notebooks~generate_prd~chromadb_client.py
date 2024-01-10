from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from llm import LLM


class ChromaDBClient:
    """Client for ChromaDB.

    Args:
        chromadb (Chroma): ChromaDB instance.
    """

    def __init__(self, chromadb: Chroma):
        # self.chromadb = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="chromadb")
        self.chromadb = chromadb
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0)

    def store_web_content(self, content: list(tuple([str, str, str]))) -> None:
        """Store web content in ChromaDB.

        Args:
            content (list(tuple(str, str, str))): List of webpages. 
            Each webpage is a tuple of (title, link, content).
            content is a string of all the paragraphs (<p></p>) in the webpage with 100+ characters.
        """

        for title, link, webpage in content:
            print(f"Storing {title}")
            doc = self.text_splitter.create_documents(
                texts=[webpage],
                metadatas=[{"source": link, "title": title}]
            )
            ids = self.chromadb.add_documents(documents=[*doc])
            print(f"Stored {title} with {len(ids)} chunks")

        return None

    def update_qa_chain(self, k: int = 2):
        """Create QA chain.

        Args:
            k (int, optional): Number of documents to retrieve. Defaults to 2.
        """
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=LLM(chat_model="openai-gpt-3.5").llm,
            retriever=self.chromadb.as_retriever(search_kwargs={"k": k}),
            chain_type="stuff",
            return_source_documents=True,
        )

    def ask_with_source(self, query: str) -> tuple([str, list]):
        """Ask question and return source documents.

        Args:
            query (str): Query string.

        Returns:
            tuple(str, list): Tuple of (answer, source_documents).
            answer is a string. 
            source_documents is a list of documents. Each document has attributes: (text, metadata). Can be accessed as document[i].text and document[i].metadata
            document[i].metadata is a dictionary with keys: source, title. Can be accessed as document[i].metadata["source"] contains the link and document[i].metadata["title"]
        """
        self.update_qa_chain(k=2)
        answer = None
        source_documents = None
        try:
            response = self.qa_chain({
                "question": query,
                "chat_history": [],
            })
        except Exception as e:
            print(e)
            return answer, source_documents

        answer = response["answer"]
        source_documents = response["source_documents"]

        return answer, source_documents
