from repolya._log import logger_rag

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from langchain.retrievers import (
    BM25Retriever,
    EnsembleRetriever,
    ParentDocumentRetriever,
    ContextualCompressionRetriever,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

from langchain.document_transformers import (
    LongContextReorder,
    EmbeddingsRedundantFilter,
    EmbeddingsClusteringFilter,
)

import faiss
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings

from typing import List
from pydantic import BaseModel, Field
import os

from repolya.rag.doc_splitter import get_RecursiveCharacterTextSplitter
from repolya.rag.embedding import get_embedding_HuggingFace
from repolya.local.textgen import get_textgen_llm


##### Multi Query Retriever
def get_vdb_multi_query_retriever(_vdb):
    class LineList(BaseModel):
        # "lines" is the key (attribute name) of the parsed output
        lines: List[str] = Field(description="Lines of text")

    class LineListOutputParser(PydanticOutputParser):
        def __init__(self) -> None:
            super().__init__(pydantic_object=LineList)
        def parse(self, text: str) -> LineList:
            lines = text.strip().split("\n")
            return LineList(lines=lines)
    output_parser = LineListOutputParser()
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions seperated by newlines.
Original question: {question}""",
    )
    llm = ChatOpenAI(
        model_name=os.getenv('OPENAI_API_MODEL'),
        temperature=0
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=QUERY_PROMPT,
        output_parser=output_parser,
    )
    _base_retriever = _vdb.as_retriever(search_kwargs={"k": 5})
    ##### Remove redundant results from the merged retrievers
    _filter = EmbeddingsRedundantFilter(embeddings=OpenAIEmbeddings())
    ##### Re-order results to avoid performance degradation
    _reordering = LongContextReorder()
    ##### ContextualCompressionRetriever
    _pipeline = DocumentCompressorPipeline(transformers=[_filter, _reordering])
    _compression_retriever_reordered = ContextualCompressionRetriever(
        base_compressor=_pipeline,
        base_retriever=_base_retriever
    )
    ##### MultiQueryRetriever
    _multi_retriever = MultiQueryRetriever(
        retriever=_compression_retriever_reordered,
        llm_chain=llm_chain,
        parser_key="lines"
    )
    return _multi_retriever


def get_vdb_multi_query_retriever_textgen(_vdb, _textgen_url):
    class LineList(BaseModel):
        # "lines" is the key (attribute name) of the parsed output
        lines: List[str] = Field(description="Lines of text")

    class LineListOutputParser(PydanticOutputParser):
        def __init__(self) -> None:
            super().__init__(pydantic_object=LineList)
        def parse(self, text: str) -> LineList:
            lines = text.strip().split("\n")
            return LineList(lines=lines)
    output_parser = LineListOutputParser()
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""### Human:
你是一名AI语言模型助手。
你的任务是针对给定问题生成3个不同的版本，用于从向量中检索相关文档数据库。
通过将给定问题生成多个变体，帮助用户克服基于距离的相似性搜索的一些限制。
仅输出这些替代性问题，并用换行符分隔，务必不用重复。

问题:
{question}

### Assistant:
""",
    )
    llm = get_textgen_llm(
        _textgen_url,
        _top_p=0.5,
        _max_tokens=5000,
        _stopping_strings=[]
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=QUERY_PROMPT,
        output_parser=output_parser,
    )
    _base_retriever = _vdb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, 'fetch_k': 20},
        # search_type="similarity_score_threshold",
        # search_kwargs={'score_threshold': 0.5},
    )
    ##### Remove redundant results from the merged retrievers
    _model_name, _embedding = get_embedding_HuggingFace()
    _filter = EmbeddingsRedundantFilter(embeddings=_embedding)
    _pipeline = DocumentCompressorPipeline(transformers=[_filter])
    # ##### Re-order results to avoid performance degradation
    # _reordering = LongContextReorder()
    # _pipeline = DocumentCompressorPipeline(transformers=[_filter, _reordering])
    ##### ContextualCompressionRetriever
    _compression_retriever_reordered = ContextualCompressionRetriever(
        base_compressor=_pipeline,
        base_retriever=_base_retriever
    )
    ##### MultiQueryRetriever
    _multi_retriever = MultiQueryRetriever(
        retriever=_compression_retriever_reordered,
        llm_chain=llm_chain,
        parser_key="lines"
    )
    return _multi_retriever


##### Ensemble Retriever
def get_docs_ensemble_retriever(_docs):
    # initialize the bm25 retriever and faiss retriever
    bm25_retriever = BM25Retriever.from_texts(_docs)
    bm25_retriever.k = 5
    embedding = OpenAIEmbeddings()
    faiss_vectorstore = FAISS.from_texts(_docs, embedding)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})
    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever


##### Parent Document Retriever
def monkeypatch_FAISS(embeddings_model):
    from typing import Iterable, List, Optional, Any
    def _add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> List[str]:
            """Run more texts through the embeddings and add to the vectorstore.
            Args:
                texts: Iterable of strings to add to the vectorstore.
                metadatas: Optional list of metadatas associated with the texts.
                ids: Optional list of unique IDs.

            Returns:
                List of ids from adding the texts into the vectorstore.
            """
            embeddings = embeddings_model.embed_documents(texts)
            return self._FAISS__add(texts, embeddings, metadatas=metadatas, ids=ids)
    FAISS.add_texts = _add_texts

def get_docs_parent_retriever(_docs):
    ### This text splitter is used to create the parent documents
    parent_splitter = get_RecursiveCharacterTextSplitter(text_chunk_size=2000, text_chunk_overlap=200)
    ### The storage layer for the parent documents
    _docstore = InMemoryStore()
    ### This text splitter is used to create the child documents
    child_splitter = get_RecursiveCharacterTextSplitter(text_chunk_size=200, text_chunk_overlap=0)
    ### The vectorstore to use to index the child chunks
    monkeypatch_FAISS(OpenAIEmbeddings())
    _vectorstore = FAISS(
        OpenAIEmbeddings(),
        faiss.IndexFlatL2(1536),
        InMemoryDocstore({}),
        {}
    )
    # _vectorstore = Chroma(collection_name="split_parents", embedding_function=OpenAIEmbeddings())
    parent_retriever = ParentDocumentRetriever(
        parent_splitter=parent_splitter,
        docstore=_docstore,
        child_splitter=child_splitter,
        vectorstore=_vectorstore,
        search_kwargs={"k": 5},
    )
    parent_retriever.add_documents(_docs)
    # sub_docs = vectorstore.similarity_search("blabla")
    # print(sub_docs[0].page_content) # short text
    # retrieved_docs = retriever.get_relevant_documents("blabla")
    # print(retrieved_docs[0].page_content) # long text
    return parent_retriever

