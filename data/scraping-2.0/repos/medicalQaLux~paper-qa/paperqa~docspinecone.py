import asyncio
import os
import re
import sys
import tempfile
from dotenv import load_dotenv
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Set, Union, cast
import pandas as pd
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')

from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.vectorstores import FAISS, VectorStore, Pinecone
from langchain.docstore.document import Document
from langchain.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance
from pydantic import BaseModel, validator

from .docs import Docs
from .chains import get_score, make_chain
from .paths import PAPERQA_DIR
from .readers import read_doc
from .types import Answer, CallbackFactory, Context, Doc, DocKey, PromptCollection, Text
from .utils import (
    gather_with_concurrency,
    guess_is_4xx,
    maybe_is_html,
    maybe_is_pdf,
    maybe_is_text,
    md5sum,
    name_in_text,
)
import pinecone

pinecone.init(
    api_key=pinecone_api_key,  # find at app.pinecone.io
    environment=pinecone_env,  # next to api key in console
)

class DocsPineCone(Docs):
    """A collection of documents to be used for answering questions."""
    doc_index_name: Optional[str] = None
    text_index_name: Optional[str] = None
    embedding_function = OpenAIEmbeddings(client=None).embed_query
    text_index_p: Optional[pinecone.Index] = None
    marginal_relevance = True

    def __init__(self, text_index_name="paperqa-index", parquet_file = '', type = 'default', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_index_name: Optional[str] = text_index_name

        self.text_index_p = pinecone.Index(text_index_name)
        if type == 'default':
            self.doc_index = Pinecone(index=self.text_index_p,text_key="text", embedding_function=self.embedding_function, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,namespace="docs")
            self.texts_index = Pinecone(index=self.text_index_p,text_key="text" ,embedding_function=self.embedding_function, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,namespace="texts")
            self.__instantiate_docs__(parquet_file)

        elif type == 'twitter':
            self.doc_index = Pinecone(index=self.text_index_p,text_key="text", embedding_function=self.embedding_function, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,namespace="docstwitter")
            self.texts_index = Pinecone(index=self.text_index_p,text_key="text" ,embedding_function=self.embedding_function, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,namespace="textstwitter")

    def __instantiate_docs__(self, parquet_file):
        #temporary hack to set up the docs object
        if not parquet_file:
            print ("WARNING: No doc parquet file detected, you need to add documents or this index will not work")
            return 
        df_docs = pd.read_parquet(parquet_file)
        for _, row in df_docs.iterrows():
            self.docs[row['dockey']] =  Doc(docname = row['docname'], citation=row['citation'], dockey = row['dockey'])
        return

    def add_texts(
        self,
        texts: List[Text],
        doc: Doc,
    ) -> bool:
        """Add chunked texts to the collection. This is useful if you have already chunked the texts yourself.

        Returns True if the document was added, False if it was already in the collection.
        """
        # if doc.dockey in self.docs:
        #     return False
        if len(texts) == 0:
            raise ValueError("No texts to add.")
        if doc.docname in self.docnames:
            new_docname = self._get_unique_name(doc.docname)
            for t in texts:
                t.name = t.name.replace(doc.docname, new_docname)
            doc.docname = new_docname

        if self.texts_index is not None:
            # try:
            # metadatas=[doc.dict()]*len(texts)
            metadatas=[ {**{"name":i.name},**i.doc.dict()} for i in texts]
            texts = [ i.text for i in texts]
            self.texts_index.add_texts( 
                texts,metadatas=metadatas
            )
        
        if self.doc_index is not None:
            self.doc_index.add_texts([doc.citation], metadatas=[doc.dict()])
        self.docs[doc.dockey] = doc
        self.texts += texts
        self.docnames.add(doc.docname)
        return True

    async def adoc_match(
        self, query: str, k: int = 25, get_callbacks: CallbackFactory = lambda x: None
    ) -> Set[DocKey]:
        """Return a list of dockeys that match the query."""
        if self.doc_index is None:
            if len(self.docs) == 0:
                return set()
            texts = [doc.citation for doc in self.docs.values()]
            metadatas = [d.dict() for d in self.docs.values()]
            documents_conv = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
            self.doc_index = Pinecone(index=self.text_index_p,text_key="text", embedding_function=self.embedding_function, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,namespace="docs")
        
         
        matches = self.doc_index.max_marginal_relevance_search(
            query, k=k + len(self.deleted_dockeys)
        )
        # filter the matches
        matches = [
            m for m in matches if m.metadata["dockey"] not in self.deleted_dockeys
        ]
        try:
            # for backwards compatibility (old pickled objects)
            matched_docs = [self.docs[m.metadata["dockey"]] for m in matches]
        except KeyError:
            matched_docs = [Doc(**m.metadata) for m in matches]
        if len(matched_docs) == 0:
            return set()
        chain = make_chain(
            self.prompts.select, cast(BaseLanguageModel, self.llm), skip_system=True
        )
        papers = [f"{d.docname}: {d.citation}" for d in matched_docs]
        result = await chain.arun(  # type: ignore
            question=query, papers="\n".join(papers), callbacks=get_callbacks("filter")
        )
        return set([d.dockey for d in matched_docs if d.docname in result])


    def __setstate__(self, state):
        object.__setattr__(self, "__dict__", state["__dict__"])
        object.__setattr__(self, "__fields_set__", state["__fields_set__"])
        try:
            self.texts_index = Pinecone.from_existing_index(self.text_index_name, self.embeddings, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,namespace="texts")
        except Exception:
            try:
                index = pinecone.Index(self.text_index_name)
                self.texts_index = Pinecone(index, embeddings.embed_query, "text",namespace="texts")
            except Exception:
                # they use some special exception type, but I don't want to import it
                self.texts_index = None
        self.doc_index = None

    def _build_texts_index(self, keys: Optional[Set[DocKey]] = None):
        if keys is not None and self.jit_texts_index:
            del self.texts_index
            self.texts_index = None
        if self.texts_index is None:
            texts = self.texts
            if keys is not None:
                texts = [t for t in texts if t.doc.dockey in keys]
            if len(texts) == 0:
                return
            raw_texts = [t.text for t in texts]
            metadata=[{"doc_id": t.doc.dockey, "docname": t.doc.docname} for t in texts ]
            documents_conv = [Document(page_content=text, metadata=metadata) for text in raw_texts]
            self.texts_index = Pinecone.from_documents(documents_conv, self.embeddings, index_name=self.text_index_name,namespace="texts")


    async def aget_evidence(
        self,
        answer: Answer,
        k: int = 5,  # Jeffrey: changed from 20
        max_sources: int = 10,  # Jeffrey: changed from 20
        marginal_relevance: bool = True,
        get_callbacks: CallbackFactory = lambda x: None,
    ) -> Answer:
        if len(self.docs) == 0 and self.doc_index is None:
            return answer
        self._build_texts_index(keys=answer.dockey_filter)
        if self.texts_index is None:
            return answer
        self.texts_index = cast(VectorStore, self.texts_index)
        _k = k
        if answer.dockey_filter is not None:
            _k = k * 10  # heuristic
        # want to work through indices but less k
        if marginal_relevance:
            matches = self.texts_index.max_marginal_relevance_search(
                answer.question, k=_k, fetch_k=5 * _k
            )
        else:
            matches = self.texts_index.similarity_search(
                answer.question, k=_k, fetch_k=5 * _k
            )
        # ok now filter
        if answer.dockey_filter is not None:
            matches = [
                m
                for m in matches
                if m.metadata["dockey"] in answer.dockey_filter
            ]

        # check if it is deleted

        matches = [
            m
            for m in matches
            if m.metadata["dockey"] not in self.deleted_dockeys
        ]

        # check if it is already in answer
        cur_names = [c.text.name for c in answer.contexts]
        matches = [m for m in matches if m.metadata["docname"] not in cur_names]
        # now finally cut down
        matches = matches[:k]
        async def process(match):
            callbacks = get_callbacks("evidence:" + match.metadata["docname"])
            summary_chain = make_chain(
                self.prompts.summary, self.summary_llm, memory=self.memory_model
            )
            # This is dangerous because it
            # could mask errors that are important- like auth errors
            # I also cannot know what the exception
            # type is because any model could be used
            # my best idea is see if there is a 4XX
            # http code in the exception
            try:
                context = await summary_chain.arun(
                    question=answer.question,
                    citation=match.metadata["citation"],
                    summary_length=answer.summary_length,
                    text=match.page_content,
                    callbacks=callbacks,
                )
            except Exception as e:
                if guess_is_4xx(str(e)):
                    return None
                raise e
            if "not applicable" in context.lower():
                return None
            
            c = Context(
                context=context,
                text=Text(
                    text=match.page_content,
                    name=match.metadata["docname"],
                    doc=Doc(docname=match.metadata["docname"], citation=match.metadata["citation"], dockey=match.metadata["dockey"]),
                ),
                score=get_score(context),
            )
            return c

        results = await gather_with_concurrency(
            self.max_concurrent, *[process(m) for m in matches]
        )
        # filter out failures
        contexts = [c for c in results if c is not None]
        if len(contexts) == 0:
            return answer
        contexts = sorted(contexts, key=lambda x: x.score, reverse=True)
        contexts = contexts[:max_sources]
        # add to answer contexts
        answer.contexts += contexts
        context_str = "\n\n".join(
            [f"{c.text.name}: {c.context}" for c in answer.contexts]
        )
        valid_names = [c.text.name for c in answer.contexts]
        context_str += "\n\nValid keys: " + ", ".join(valid_names)
        answer.context = context_str
        return answer