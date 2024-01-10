"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import os.path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Iterable, Any, cast

from pyspark.sql import SparkSession, DataFrame
from ray.data import Dataset

from pyrecdp.core.import_utils import check_availability_and_install, import_sentence_transformers
from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS
from pyrecdp.primitives.operations.logging_utils import logger


def create_embeddings(embeddings_cls: Optional[str] = None, embeddings_construct_args: Optional[dict[str, Any]] = None):
    """currently we only use langchain embeddings"""
    if embeddings_cls is None:
        embeddings_cls = 'HuggingFaceEmbeddings'

    if embeddings_cls in ['HuggingFaceEmbeddings', 'HuggingFaceInstructEmbeddings', 'HuggingFaceBgeEmbeddings']:
        import_sentence_transformers()

    if embeddings_cls == 'HuggingFaceInstructEmbeddings':
        check_availability_and_install("InstructorEmbedding")

    from pyrecdp.core.class_utils import new_instance
    embeddings_construct_args = embeddings_construct_args or {}
    embeddings = new_instance('langchain.embeddings', embeddings_cls, **embeddings_construct_args)
    from langchain.schema.embeddings import Embeddings
    assert isinstance(embeddings, Embeddings)
    return embeddings


class DocumentStore(ABC):
    """interface for vector store"""

    def __init__(self, text_column: str,
                 embeddings_column: Optional[str] = 'embedding',
                 embeddings: Optional[str] = None,
                 embeddings_args: Optional[Dict] = None,
                 vector_store_args: Optional[Dict] = None,
                 override: bool = False):
        self.text_column = text_column
        self.embeddings = embeddings
        self.embeddings_column = embeddings_column
        self.vector_store_args = vector_store_args
        self.override = override
        self.embeddings_args = embeddings_args or {}

    def is_vector_store(self):
        return True

    def persist(self, ds: Union[Dataset, DataFrame]):
        """interface for persist embeddings to underlying vector store"""

        if self.is_vector_store():
            check_availability_and_install(["langchain"])
            if isinstance(ds, Dataset):
                ds = self.embedding_with_ray(ds)
            else:
                ds = self.embedding_with_spark(ds)

        db = self.do_persist(ds)
        if self.vector_store_args["return_db_handler"]:
            return db
        else:
            return ds

    @abstractmethod
    def do_persist(self, ds: Union[Dataset, DataFrame]):
        """base interface for vector store to persist the text and embeddings"""

    def embedding_with_spark(self, df: DataFrame, **kwargs):
        import pandas as pd
        from pyspark.sql import types as T
        def batch_embedding(batches: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            lc_embedding = create_embeddings(self.embeddings, self.embeddings_args)
            for pdf in batches:
                pdf[self.embeddings_column] = lc_embedding.embed_documents(pdf[self.text_column])
                yield pdf

        fields = [field for field in df.schema] + [T.StructField(self.embeddings_column, T.ArrayType(T.FloatType()))]
        df = df.mapInPandas(batch_embedding, T.StructType(fields))
        return df

    def embedding_with_ray(self, ds: Dataset):

        def batch_embedding(batch, text_column: str, embedding_column: str, embeddings: str,
                            embedding_kwargs: Optional[dict[str, Any]] = None):
            lc_embedding = create_embeddings(embeddings, embedding_kwargs)
            batch[embedding_column] = lc_embedding.embed_documents(batch[text_column])
            return batch

        ds = ds.map_batches(
            lambda batch: batch_embedding(
                batch,
                self.text_column,
                self.embeddings_column,
                self.embeddings,
                self.embeddings_args
            ),
            batch_format='pandas',
        )
        return ds


class EmbeddingsOnlyStore(DocumentStore):

    def do_persist(self, ds: Union[Dataset, DataFrame], **kwargs):
        return ds


class LangchainFAAIS(DocumentStore):
    def do_persist(self, ds: Dataset):
        check_availability_and_install(["langchain", "faiss-cpu"])

        db = self.vector_store_args["db_handler"]
        index_name = self.vector_store_args.get("index", "index")

        rows = ds.iter_rows() if isinstance(ds, Dataset) else ds.collect()
        text_embeddings = [(row[self.text_column], row[self.embeddings_column]) for row in rows]
        if not bool(text_embeddings):
            logger.error("Text embeddings is empty, no data to store!")
            return db
        from langchain.vectorstores.faiss import FAISS
        if db is not None:
            db.add_embeddings(text_embeddings)
            return db

        if "output_dir" not in self.vector_store_args:
            raise ValueError(f"You must have `output_dir` option specify for FAAIS vector store")
        faiss_folder_path = self.vector_store_args["output_dir"]
        embeddings = create_embeddings(self.embeddings, self.embeddings_args)
        if not self.override and os.path.exists(os.path.join(faiss_folder_path, index_name + ".faiss")):
            db = FAISS.load_local(faiss_folder_path, embeddings, index_name)
            db.add_embeddings(text_embeddings)
        else:
            db = FAISS.from_embeddings(text_embeddings, embedding=embeddings)

        db.save_local(faiss_folder_path, index_name)
        return db


class LangchainChroma(DocumentStore):
    def persist(self, ds):
        db = self.do_persist(ds)
        if self.vector_store_args["return_db_handler"]:
            return db
        else:
            return ds

    def do_persist(self, ds):
        check_availability_and_install(["chromadb==0.4.15", "langchain"])
        chroma = self.vector_store_args["db_handler"]

        collection_name = self.vector_store_args.get("collection_name", 'langchain')
        rows = ds.iter_rows() if isinstance(ds, Dataset) else ds.collect()
        texts = [row[self.text_column] for row in rows]

        from langchain.vectorstores.chroma import Chroma
        if chroma is not None:
            chroma.add_texts(texts)
            return chroma
        if "output_dir" not in self.vector_store_args and 'persist_directory' not in self.vector_store_args:
            raise ValueError(
                f"You must have `output_dir` or `persist_directory` option specify for Chroma vector store")

        if 'output_dir' in self.vector_store_args:
            persist_directory = self.vector_store_args["output_dir"]
        else:
            persist_directory = self.vector_store_args["persist_directory"]

        embeddings = create_embeddings(self.embeddings, self.embeddings_args)
        if not self.override and os.path.exists(persist_directory):
            chroma = Chroma(collection_name=collection_name,
                            persist_directory=persist_directory,
                            embedding_function=embeddings)
            chroma.add_texts(texts)
        else:
            chroma = Chroma.from_texts(texts,
                                       collection_name=collection_name,
                                       embedding=embeddings,
                                       persist_directory=persist_directory)
        chroma.persist()
        return chroma


class HaystackElasticSearch(DocumentStore):

    def is_vector_store(self):
        return False

    def do_persist(self, ds):
        check_availability_and_install(["farm-haystack", "farm-haystack[elasticsearch7]"])
        exclude_keys = ['db_handler', 'return_db_handler']
        vector_store_args = dict((k, v) for k, v in self.vector_store_args.items() if k not in exclude_keys)
        if isinstance(ds, Dataset):
            def batch_index(batch, text_column, vector_store_args: Optional[Dict[str, Any]]):
                from haystack.document_stores import ElasticsearchDocumentStore
                elasticsearch = ElasticsearchDocumentStore(
                    **vector_store_args
                )
                from haystack import Document as SDocument
                documents = [SDocument(content=text) for text in batch[text_column]]
                elasticsearch.write_documents(documents)

                return {}

            ds.map_batches(lambda batch: batch_index(batch, self.text_column, vector_store_args)).count()
        else:
            def batch_index_with_var(batch, bv_value):
                from haystack import Document as SDocument
                text_column, vector_store_args = bv_value.value
                from haystack.document_stores import ElasticsearchDocumentStore
                elasticsearch = ElasticsearchDocumentStore(
                    **vector_store_args
                )
                documents = [SDocument(content=row[text_column]) for row in batch]
                elasticsearch.write_documents(documents)

            ds = cast(DataFrame, ds)

            bv = ds.sparkSession.sparkContext.broadcast((self.text_column, vector_store_args))
            ds.foreachPartition(lambda p: batch_index_with_var(p, bv))

        # share this document store only when rag retrieval want to use document store created from index stage
        elasticsearch = self.vector_store_args["db_handler"]
        if elasticsearch is None:
            from haystack.document_stores import ElasticsearchDocumentStore
            elasticsearch = ElasticsearchDocumentStore(
                **vector_store_args
            )
        return elasticsearch


class DocumentIngestion(BaseLLMOperation):
    def __init__(self,
                 text_column: str = 'text',
                 embeddings_column: Optional[str] = 'embedding',
                 embeddings: Optional[str] = None,
                 embeddings_args: Optional[dict] = None,
                 vector_store: Optional[str] = None,
                 vector_store_args: Optional[dict] = None,
                 override: bool = False,
                 return_db_handler=False,
                 db_handler=None,
                 requirements=None,
                 **kwargs):
        """
          Document ingestion operator.
          Args:
            text_column: The name of the column containing the text data.
            rag_framework: The RAG framework to use. The default is 'langchain'.
            embeddings_column: The name of the column to store the embeddings.
            embeddings: The type of embeddings to use.
                     
            embeddings_args: Optional arguments for the embeddings. Examples: 'OpenAIEmbeddings', 'HuggingFaceEmbeddings'.
                      If the embeddings property is specified, then the documents and their embeddings will be written to the vector database.
            vector_store: The type of vector store or document store to use. Current we support 'faiss' and 'chroma' for vector store, and 'elasticsearch' for document store.
            vector_store_args: Optional arguments for the vector store.
            override: Whether to override the existing embeddings and vector store.
            return_db_handler: If false, return dataset; If True, return created vectorDB handler.
            db_handler: Use pre-created db_handler as input.
            """
        if requirements is None:
            requirements = []

        if text_column is None:
            raise ValueError(f"text column is required")

        if vector_store is None:
            raise ValueError(f"vector store is required")

        settings = {
            'text_column': text_column,

            'embeddings_column': embeddings_column,
            'embeddings': embeddings,
            'embeddings_args': embeddings_args,

            'vector_store': vector_store,
            'vector_store_args': vector_store_args,
            'override': override,
            'requirements': requirements,

            'return_db_handler': return_db_handler,
            'db_handler': db_handler,
        }
        requirements = requirements
        super().__init__(settings, requirements)
        self.support_ray = True
        self.support_spark = True
        self.text_column = text_column
        self.embeddings_column = embeddings_column,
        self.embeddings = embeddings
        self.embeddings_args = embeddings_args or {}

        self.vector_store = vector_store.lower()
        self.vector_store_args = vector_store_args or {}
        self.override = override
        self.embeddings_column = embeddings_column
        self.document_store = self._create_document_store()
        self.vector_store_args['return_db_handler'] = return_db_handler
        self.vector_store_args['db_handler'] = db_handler

    def _create_document_store(self) -> DocumentStore:
        document_store_ctor_args = {
            'text_column': self.text_column,
            'embeddings_column': self.embeddings_column,
            'embeddings': self.embeddings,
            'embeddings_args': self.embeddings_args,
            'vector_store_args': self.vector_store_args,
            'override': self.override,
        }

        if not self.embeddings and not self.vector_store:
            return EmbeddingsOnlyStore(**document_store_ctor_args)

        if self.embeddings:
            if 'faiss' == self.vector_store:
                return LangchainFAAIS(**document_store_ctor_args)
            elif 'chroma' == self.vector_store:
                return LangchainChroma(**document_store_ctor_args)
            else:
                raise NotImplementedError(
                    f"vector store {self.vector_store} is not supported yet!")
        else:
            if 'elasticsearch' == self.vector_store:
                return HaystackElasticSearch(**document_store_ctor_args)
            else:
                raise NotImplementedError(
                    f"document store {self.vector_store} is not supported yet!")

    def process_rayds(self, ds: Dataset = None):
        return self.document_store.persist(ds)

    def process_spark(self, spark: SparkSession, df: DataFrame = None):
        return self.document_store.persist(df)


LLMOPERATORS.register(DocumentIngestion)
