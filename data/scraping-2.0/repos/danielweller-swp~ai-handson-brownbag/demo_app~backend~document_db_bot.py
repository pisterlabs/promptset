from langchain.vectorstores import VectorStore
from typing import Any, Iterable, List, Optional

from openai.embeddings_utils import get_embedding, cosine_similarity

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import AzureOpenAI
import openai
import os
from azure_openai import setupAzureOpenAI
import json
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from utils import get_html_page_text

gpt35 = setupAzureOpenAI()

def get_embedding(text):
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(
       api_type="open_ai",
       api_base="https://api.openai.com/v1",
       api_version="2020-11-07",
       api_key=os.getenv("OPENAI_OPENAI_API_KEY"),
       input=[text], 
       engine='text-embedding-ada-002')["data"][0]["embedding"]   

def create_embedding(text, metadata):
  embedding = get_embedding(text)
  return {
     "text": text,     
     "embedding": embedding,
     "metadata": metadata
  }

class MyDocument(Document):
    def __init__(self, page_content, metadata) -> None:
       self.page_content = page_content
       if metadata is None:
          self.metadata = dict()
       else:
          self.metadata = metadata

class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
       self.embeddings = []
       super().__init__()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if metadatas is None:
          text_embeddings = [create_embedding(text, None) for text in texts]
        else:
          text_embeddings = [create_embedding(text, md) for (text, md) in zip(texts, metadatas)]
        self.embeddings.extend(text_embeddings)


       
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
       q_embedding = get_embedding(query)
       matches = sorted(self.embeddings, key=lambda e: cosine_similarity(e["embedding"], q_embedding), reverse=True)
       results = [Document(page_content=e["text"], metadata=e["metadata"]) for e in matches[:k]]
       return results

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        result = InMemoryVectorStore()
        result.add_texts(texts, metadatas)
        return result


def setupVectorStore():
  vectorstore = InMemoryVectorStore()
  sources = [
     "https://trustbit.tech/our-expertise",
     "https://trustbit.tech/our-culture",
     "https://trustbit.tech/de/impressum"
  ]
  texts = [get_html_page_text(s) for s in sources]
  vectorstore.add_texts(texts, [{"source": s} for s in sources])
  return vectorstore


vectorstore = setupVectorStore()

qa = load_qa_with_sources_chain(gpt35, chain_type="stuff")

def get_document_db_bot_completion(prompt, retries=0):
  docs = vectorstore.similarity_search(prompt, 1)
  try:
    result = qa(
       {"input_documents": docs, "question": prompt, "chat_history": []},
       return_only_outputs=True)
    return result["output_text"]
  except Exception as e:
     if retries < 5:
        return get_document_db_bot_completion(prompt, retries+1)
     else:
        raise e
