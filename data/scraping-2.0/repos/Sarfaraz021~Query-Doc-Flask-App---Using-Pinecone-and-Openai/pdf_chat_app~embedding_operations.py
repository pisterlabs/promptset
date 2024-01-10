from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from document import Document  # Adjust this import based on the actual module structure
import pinecone

class EmbeddingOperations:
    def __init__(self, texts):
        self.index_name = "pdf_chat_index"
        self.embeddings = "text_embeddings"

        # Check if texts are already Document objects
        if not all(isinstance(text, Document) for text in texts):
            # If not, create Document objects
            texts = [Document(page_content=text) if isinstance(text, str) else text for text in texts]

        # Now, texts should be a list of Document objects
        pinecone.init(api_key="", environment="")
        self.docsearch = pinecone.Index(index=self.index_name, embedding=self.embeddings, text_key="page_content")

        self.docsearch.create_index(dimension=300, metric="cosine")

        # self.docsearch.index([str(t.page_content) for t in texts])
        self.docsearch.upsert([{"page_content": str(t.page_content)} for t in texts])

