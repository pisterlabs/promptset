#!/usr/bin/env python3
import functools
import hashlib
import sys
from typing import TypeVar

import loguru
import polars as pl
from beartype import beartype
from datasets import load_dataset
from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch
from langchain.schema import Document
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import VectorStore, Qdrant
from qdrant_client import QdrantClient

from indexpaper.resolvers import *
from indexpaper.utils import timing

T = TypeVar('T')

DEFAULT_COLUMNS = ('corpusid',
                  'content_source_oainfo_openaccessurl',
                  'updated',
                  'externalids_doi',
                  'externalids_pubmed',
                  'annotations_abstract',
                  'annotations_author',
                  'annotations_title',
                   'annotations_paragraph',
                   #'content_text'
                   )



class Paperset:
    """
    The class that makes it easier to work with dataset or papers for indexing, can be either hugging face or parquet
    """

    @staticmethod
    def generate_id_from_data(data):
        """
        function to avoid duplicates
        :param data:
        :return:
        """
        if isinstance(data, str):  # check if data is a string
            data = data.encode('utf-8')  # encode the string into bytes
        return str(hex(int.from_bytes(hashlib.sha256(data).digest()[:32], 'little')))[-32:]

    @staticmethod
    @beartype
    def default_transform(content: list[str], size: int = 10, step: int = 10) -> list[str]:
        result = content if len(content) <2 else seq(content).sliding(size, step).map(lambda s: s.reduce(lambda x, y: x + "\n" + y)).to_list()
        #print(f"**************************************************************\nfrom {len(content)} to {len(result)}")
        return result


    lazy_frame: pl.LazyFrame
    content_field: str #"content_text" #'annotations_paragraph'
    splitter: Optional[TextSplitter] = None
    columns: list[str]


    @staticmethod
    def get_dataset(name: str, default_columns: Optional[list|tuple] = DEFAULT_COLUMNS) -> pl.LazyFrame:
        """
        for example "longevity-genie/moskalev_papers"
        :param name:
        :return: polars Dataframe
        """
        dataset = load_dataset(name)["train"]
        df = pl.from_arrow(dataset.data.table).lazy()
        return df if default_columns is None else df.select(default_columns)

    transform_content: Optional[Callable[[list], list]]

    @beartype
    def __init__(self, df_name_or_path: Union[pl.LazyFrame, str, Path],
                 splitter: Optional[TextSplitter] = None,
                 content_field: str = "annotations_paragraph",#'content_text',
                 default_columns=DEFAULT_COLUMNS, low_memory: bool = False,
                 transform_content: Optional[Callable[[list], list]] = None,
                 paragraphs_together: int = 5
                 ):
        self.splitter = splitter
        if isinstance(df_name_or_path, pl.LazyFrame):
            self.lazy_frame = df_name_or_path if default_columns is None else df_name_or_path.select(default_columns)
        elif isinstance(df_name_or_path, Path) or "parquet" in df_name_or_path:
            df = pl.scan_parquet(df_name_or_path, low_memory=low_memory)
            self.lazy_frame = df if default_columns is None else df.select(default_columns)
        else:
            self.lazy_frame = Paperset.get_dataset(df_name_or_path, default_columns)
        self.content_field = content_field
        self.columns = self.lazy_frame.columns
        self.transform_content = functools.partial(self.default_transform, size = paragraphs_together, step = paragraphs_together) if transform_content is None else transform_content
        assert content_field in self.columns, f"{content_field} has not been found in dataframe columns {self.columns}"

    @beartype
    def split_documents(self, documents: list[Document])-> list[Document]:
        """
        splits documents of splitter is present
        :param documents:
        :return:
        """
        return documents if self.splitter is None else self.splitter.split_documents(documents)

    @beartype
    def row_to_documents(self, row: tuple):
        """
        Converts dataframe row into the text for indexing
        :param row:
        :param columns:
        :param content_field:
        :param splitter:
        :return:
        """
        d: dict = dict(zip(self.columns, row))
        data = d[self.content_field]
        contents = self.transform_content(data if type(data) is list else [data])
        meta = {k:v for k,v in d.items() if k != self.content_field}
        def with_index(meta: dict, i: int):
            meta["paragraph"] = i
            if "doi" in meta and meta["doi"] is not None:
                meta["source"] = meta["doi"] + "#" + str(i)
            elif "externalids_doi" in meta and meta["externalids_doi"] is not None:
                meta["doi"] = meta["externalids_doi"]
                meta["source"] = meta["doi"] + "#" + str(i)
            elif "externalids_pubmed" in meta and meta["externalids_pubmed"] is not None:
                meta["source"] = str(meta["externalids_pubmed"]) + "#" + str(i)
            elif "corpusid" in meta and meta["corpusid"] is not None:
                meta["source"] = str(meta["corpusid"]) + "#" + str(i)
            else:
                meta["source"] = "unspecified" + "#" + str(i)
            return meta
        docs: list[Document] = [Document(page_content = c, metadata=with_index(meta, i)) for (c, i) in seq(contents).zip_with_index(1)]
        return self.split_documents(docs)

    @beartype
    def documents_from_dataset_slice(self, df: pl.DataFrame) -> list[Document]:
        """
        turns slices of n papers into documents by calling row_to_documeents(r) on each paper
        :param df:
        :return:
        """
        return seq(self.row_to_documents(r) for r in df.iter_rows()).flatten().to_list()


    def fold_left_slices(self, n: int, fold: Callable[[T, pl.DataFrame], T], acc: T, start: int = 0) -> T:
        """
        Function that simulates fold_left on slices of lazy dataframe
        :param df:
        :param n:
        :param fold:
        :param acc:
        :param start:
        :return:
        """
        # Get the slice
        slice_lazy_df = self.lazy_frame.slice(start, n)

        # Collect the slice to a DataFrame to check if it has zero rows
        slice_df = slice_lazy_df.collect()
        if slice_df.shape[0] == 0:
            return acc

        # Apply the function to the slice (in place modification)
        upd_acc = fold(acc, slice_df)

        # Recursive call to process the next slice
        return self.fold_left_slices(n, fold, upd_acc, start + n)

    def fold_left_document_slices(self, n: int, fold: Callable[[T, list[Document]], T], acc: T, start: int = 0) -> T:
        """
        Wrapper to apply it to the documents
        :param n: how many papers to take in a slice
        :param fold:
        :param acc:
        :param start:
        :return:
        """
        def fold_df(value: T, df: pl.DataFrame) -> T:
            return fold(value, self.documents_from_dataset_slice(df))
        return self.fold_left_slices(n, fold_df, acc, start)

    def foreach_numbered_slice(self, n: int, fun: Callable[[pl.DataFrame, int, int], None], start: int = 0) -> None:
        # Get the slice
        slice_lazy_df = self.lazy_frame.slice(start, n)

        # Collect the slice to a DataFrame to check if it has zero rows
        slice_df = slice_lazy_df.collect()
        if slice_df.shape[0] == 0:
            return

        # Apply the function to the slice (in place modification)
        fun(slice_df, n, start)

        # Recursive call to process the next slice
        self.foreach_numbered_slice(n, fun, start + n)

    def foreach_slice(self, n: int, fun: Callable[[pl.DataFrame], None], start: int = 0) -> None:
        # Get the slice
        slice_lazy_df = self.lazy_frame.slice(start, n)

        # Collect the slice to a DataFrame to check if it has zero rows
        slice_df = slice_lazy_df.collect()
        if slice_df.shape[0] == 0:
            return

        # Apply the function to the slice (in place modification)
        fun(slice_df)

        # Recursive call to process the next slice
        self.foreach_slice(n, fun, start + n)

    #@beartype
    @logger.catch(reraise=True)
    def foreach_document_slice(self, n: int, fun: Callable[[list[Document]], None], start: int = 0) -> None:
        def fun_df(df: pl.DataFrame) -> None:
            return fun(self.documents_from_dataset_slice(df))
        sys.setrecursionlimit(100000) #to avoid issues
        result = self.foreach_slice(n, fun_df, start)
        sys.setrecursionlimit(5000) #putting back
        return result

    def foreach_numbered_document_slice(self, n: int, fun: Callable[[list[Document], int, int], None], start: int = 0) -> None:
        def fun_df(df: pl.DataFrame, f_n: int, f_start: int) -> None:
            return fun(self.documents_from_dataset_slice(df), f_n, f_start)
        return self.foreach_numbered_slice(n, fun_df, start)

    @beartype
    def fast_index_by_slice(self, n: int, client: QdrantClient, collection_name: str, batch_size: int = 32, start: int = 0, parallel: Optional[int] = None):
        @timing(f"one more slice of {n} papers has been fast-indexed")
        def fast_index_paper_slice(docs: list[Document]) -> None:
            if len(docs) == 0:
                logger.info(f"no more documents to index!")
                return None
            texts = [d.page_content for d in docs]
            for d in docs:
                if "metadata" not in d.metadata: #ugly fix for metadata issue
                    d.metadata["metadata"] = d.metadata.copy()
            metadatas = [d.metadata for d in docs]
            ids = [self.generate_id_from_data(d.page_content) for d in docs]
            client.add(
                collection_name=collection_name,
                documents=texts,
                metadata=metadatas,
                ids=ids,
                batch_size=batch_size,
                parallel=parallel
            )

        return self.foreach_document_slice(n, fast_index_paper_slice, start = start)


    def index_hybrid_by_slices(self, n: int, hybrid: OpenSearchHybridSearch, start: int = 0,
                               pipeline_name: str = "norm-pipeline",
                               logger: Optional["loguru.Logger"] = None, verbose: bool = False):
        if not hybrid.check_pipeline_exists(pipeline_name):
            logger.warning(f"hybrid pipeline does not exist, creating pipeline")
            hybrid.create_pipeline(hybrid.opensearch_url, hybrid.login, hybrid.password, pipeline_name)
        log = loguru.Logger if logger is None else logger
        @log.catch(reraise=True)
        @timing(f"one more slice of {n} papers has been indexed")
        def index_paper_slice(docs: list[Document], f_n: int, f_start: int) -> None:
            if len(docs) == 0:
                log.info(f"no more documents to index!")
                return None
            else:
                log.info(f"slice of {f_n} at {f_start} staring from {start}")
            texts = [d.page_content for d in docs]
            metadatas = [d.metadata for d in docs]
            ids = [self.generate_id_from_data(d.page_content) for d in docs]
            hybrid.add_texts(texts=texts, metadatas=metadatas, ids=ids, bulk_size = 15000)

        return self.foreach_numbered_document_slice(n, index_paper_slice, start = start)

    def index_hybrid_by_slices_detailed(self, n: int, index_name: str, embeddings: Embeddings,
                                        url: str,
                                        start: int = 0,
                                        login: Optional[str] = None,
                                        password: Optional[str] = None,
                                        pipeline_name: str = "norm-pipeline",
                                        logger: Optional["loguru.Logger"] = None):
        login = os.getenv("OPENSEARCH_USER", "admin") if login is None else login
        password = os.getenv("OPENSEARCH_PASSWORD", "admin") if password is None else password
        hybrid: OpenSearchHybridSearch = OpenSearchHybridSearch.create(url, index_name, embeddings, login=login, password=password)
        return self.index_hybrid_by_slices(n, hybrid, start, pipeline_name, logger)

    @beartype
    def index_by_slices(self, n: int, db: VectorStore, start: int = 0, logger: Optional["loguru.Logger"] = None):
        """
        :param n: number of papers included in the slice
        :param db: vector store to store results
        :param start: start index
        :param logger: logger to log data to
        :return:
        """
        log = loguru.Logger if logger is None else logger
        @timing(f"one more slice of {n} papers has been indexed")
        def index_paper_slice(docs: list[Document]) -> None:
            if len(docs) == 0:
                log.info(f"no more documents to index!")
                return None
            texts = [d.page_content for d in docs]
            metadatas = [d.metadata for d in docs]
            ids = [self.generate_id_from_data(d.page_content) for d in docs]
            db.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        return self.foreach_document_slice(n, index_paper_slice, start = start)

