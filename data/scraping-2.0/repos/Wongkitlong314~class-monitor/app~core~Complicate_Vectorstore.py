import openai
import datetime
from typing import Any, Dict, List, Optional

from langchain.schema import BaseMemory, Document
from langchain.utils import mock_now
import time
from pydantic import BaseModel, Field
import datetime
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain.schema import BaseRetriever, Document
from langchain.docstore import InMemoryDocstore
# from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
from langchain.embeddings import OpenAIEmbeddings
openai.api_key = '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'



class Document(BaseModel):
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    """String text."""
    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """


def _get_hours_passed(time: datetime.datetime, ref_time: datetime.datetime) -> float:
    """Get the hours passed between two datetime objects."""
    return (time - ref_time).total_seconds() / 3600


class Chatbot_Vectorstore(BaseModel):
    """Retriever combining embedding similarity with recency."""

    vectorstore: VectorStore
    """The vectorstore to store documents and determine salience."""

    search_kwargs: dict = Field(default_factory=lambda: dict(k=100))
    """Keyword arguments to pass to the vectorstore similarity search."""

    # TODO: abstract as a queue
    memory_stream: List[Document] = Field(default_factory=list)
    """The memory_stream of documents to search through."""

    movie_stream: List[Document] = Field(default_factory=list)
    """The movie_stream of documents to search through."""

    decay_rate: float = Field(default=0.01)
    """The exponential decay factor used as (1.0-decay_rate)**(hrs_passed)."""

    k: int = 4
    """The maximum number of documents to retrieve in a given call."""

    other_score_keys: List[str] = []
    """Other keys in the metadata to factor into the score, e.g. 'importance'."""

    default_salience: Optional[float] = None
    """The salience to assign memories not retrieved from the vector store.

    None assigns no salience to documents not fetched from the vector store.
    """
    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
    
    def _get_combined_score(
        self,
        document: Document,
        vector_relevance: Optional[float],
        current_time: datetime.datetime,
    ) -> float:
        """Return the combined score for a document."""
        hours_passed = _get_hours_passed(
            current_time,
            document.metadata["last_accessed_at"],
        )
        score = (1.0 - self.decay_rate) ** hours_passed
        for key in self.other_score_keys:
            if key in document.metadata:
                score += document.metadata[key]
        if vector_relevance is not None:
            score += vector_relevance
        return score

    def _get_combined_score_list(
        self,
        document: Document,
        vector_relevance: Optional[float],
        current_time: datetime.datetime,
    ) -> float:
        """Return the combined score for a document."""
        hours_passed = _get_hours_passed(
            current_time,
            document.metadata["last_accessed_at"],
        )
        if hours_passed < 0:
            hours_passed = 0
        score_time = (1.0 - self.decay_rate) ** hours_passed
        if score_time > 1:
            score_time = 1
        list_scores = []
        list_scores.append(score_time)
        for key in self.other_score_keys:
            if key in document.metadata:
                # score += document.metadata[key]
                list_scores.append(document.metadata[key])
        if vector_relevance is not None:
            # score += vector_relevance
            list_scores.append(1-vector_relevance)
        return list_scores

    def get_salient_docs(self, query: str) -> Dict[int, Tuple[Document, float]]:
        """Return documents that are salient to the query."""
        docs_and_scores: List[Tuple[Document, float]]
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, **self.search_kwargs
        )
        results = {}
        for fetched_doc, relevance in docs_and_scores:
            #print(fetched_doc,relevance)
            if "buffer_idx" in fetched_doc.metadata:
                buffer_idx = fetched_doc.metadata["buffer_idx"]
                doc = self.memory_stream[buffer_idx]
                results[buffer_idx] = (doc, relevance)
        return results

    def get_relevant_documents(self, query: str, current_time: Optional[Any]) -> List[Document]:
        """Return documents that are relevant to the query."""
        if current_time is None:            
            current_time = datetime.datetime.now()
        docs_and_scores = {
            doc.metadata["buffer_idx"]: (doc, self.default_salience)
            for doc in self.memory_stream[-self.k :]
        }
        # If a doc is considered salient, update the salience score
        docs_and_scores.update(self.get_salient_docs(query))
        rescored_docs = [
            (doc, self._get_combined_score_list(doc, relevance, current_time))
            for doc, relevance in docs_and_scores.values()
        ]
        score_array = [b for a,b in rescored_docs]
        score_array_np = np.array(score_array)
        delta_np = score_array_np.max(axis=0)-score_array_np.min(axis=0)
        delta_np = np.where(delta_np == 0, 1, delta_np)
        x_norm = (score_array_np-score_array_np.min(axis=0))/delta_np
        # Weight importance score less
        x_norm[:,0] = x_norm[:,0]*0.9
        x_norm[:,1] = x_norm[:,1]*0.9
        x_norm_sum = x_norm.sum(axis=1)
        rescored_docs = [
            (doc, score)
            for (doc, _), score in zip(rescored_docs,x_norm_sum)
        ]                                                     
        
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        # Ensure frequently accessed memories aren't forgotten
        for doc, _ in rescored_docs[: self.k]:
            # TODO: Update vector store doc once `update` method is exposed.
            buffered_doc = self.memory_stream[doc.metadata["buffer_idx"]]
            buffered_doc.metadata["last_accessed_at"] = current_time
            result.append(buffered_doc)
        return result

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        current_time = kwargs.get("current_time")
        if current_time is None:
            current_time = datetime.datetime.now()
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = current_time
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return self.vectorstore.add_documents(dup_docs)
    


def score_normalizer(val: float) -> float:
    return 1 - 1 / (1 + np.exp(val))



embeddings_model = OpenAIEmbeddings(openai_api_key=openai.api_key)
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=score_normalizer)
chatbot_vector = Chatbot_Vectorstore(vectorstore=vectorstore, other_score_keys=["importance"], k=1)

# initialize the elements which will be stored in the vectorstore
document1 = Document(
    page_content="select user's quiz result",
    metadata={"importance": 1, "buffer_idx":0, 'last_accessed_at':datetime.datetime.now(), 'created_at':datetime.datetime.now()},
)
document2 = Document(
    page_content="select user's records of attendence",
    metadata={"importance": 1, "buffer_idx":1, 'last_accessed_at':datetime.datetime.now(), 'created_at':datetime.datetime.now()},
)
document3 = Document(
    page_content="select user's wrting records",
    metadata={"importance": 1, "buffer_idx":2, 'last_accessed_at':datetime.datetime.now(), 'created_at':datetime.datetime.now()},
)
document4 = Document(
    page_content="select user's speaking records",
    metadata={"importance": 1, "buffer_idx":3, 'last_accessed_at':datetime.datetime.now(), 'created_at':datetime.datetime.now()},
)
document5 = Document(
    page_content="select user's reading records",
    metadata={"importance": 1, "buffer_idx":4, 'last_accessed_at':datetime.datetime.now(), 'created_at':datetime.datetime.now()},
)
document6 = Document(
    page_content="select user's interaction history of recommender system",
    metadata={"importance": 1, "buffer_idx":5, 'last_accessed_at':datetime.datetime.now(), 'created_at':datetime.datetime.now()},
)
document7 = Document(
    page_content="select user's name",
    metadata={"importance": 1, "buffer_idx":6, 'last_accessed_at':datetime.datetime.now(), 'created_at':datetime.datetime.now()},
)
document8 = Document(
    page_content="select user's gender",
    metadata={"importance": 1, "buffer_idx":7, 'last_accessed_at':datetime.datetime.now(), 'created_at':datetime.datetime.now()},
)
document9 = Document(
    page_content="select user's interests",
    metadata={"importance": 1, "buffer_idx":8, 'last_accessed_at':datetime.datetime.now(), 'created_at':datetime.datetime.now()},
)


# add the elements to the vectorstore
chatbot_vector.memory_stream.append(document1)
chatbot_vector.memory_stream.append(document2)
chatbot_vector.memory_stream.append(document3)
chatbot_vector.memory_stream.append(document4)
chatbot_vector.memory_stream.append(document5)
chatbot_vector.memory_stream.append(document6)
chatbot_vector.memory_stream.append(document7)
chatbot_vector.memory_stream.append(document8)
chatbot_vector.memory_stream.append(document9)
chatbot_vector.vectorstore.add_documents(chatbot_vector.memory_stream)

# retrieve top-1 relevant documents
result = chatbot_vector.get_relevant_documents('Check my attendence history', current_time=None)
print(result[0].page_content)

# save the vectorstore to disk
chatbot_vector.vectorstore.save_local("sql_index")

# load the vectorstore from disk to memory
new_db = FAISS.load_local("sql_index", embeddings_model)
# try retrieve top-1 relevant documents using new vectorstore loaded from index in the disk
print(new_db.similarity_search('Check my attendence history')[0].page_content)