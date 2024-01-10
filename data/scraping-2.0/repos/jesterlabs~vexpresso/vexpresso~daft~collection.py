from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Union

import daft
import pandas as pd
import pyarrow.parquet as pq
from daft import col

from vexpresso.collection import Collection
from vexpresso.daft.filter import FilterHelper
from vexpresso.daft.utils import Wrapper, indices, retrieve
from vexpresso.embedding_functions import get_embedding_fn
from vexpresso.retrievers import BaseRetriever, Retriever
from vexpresso.utils import (
    DataType,
    Document,
    ResourceRequest,
    Transformation,
    lazy,
    transformation,
)


class DaftCollection(Collection):
    def __init__(
        self,
        data: Optional[Union[str, pd.DataFrame, Dict[str, Any]]] = None,
        retriever: BaseRetriever = Retriever(),
        embeddings: Optional[List[Any]] = None,
        embedding_functions: Dict[str, Any] = {},
        daft_df: Optional[daft.DataFrame] = None,
        lazy: bool = False,
    ):
        self.daft_df = daft_df
        self.retriever = retriever
        self.embedding_functions = embedding_functions

        _metadata = {}

        if data is not None:
            if isinstance(data, str):
                if data.endswith(".json"):
                    with open(data, "r") as f:
                        _metadata = pd.DataFrame(json.load(f))
            elif isinstance(data, pd.DataFrame):
                _metadata = data.to_dict("list")
            else:
                _metadata = data

        if daft_df is None and len(_metadata) > 0:
            if isinstance(_metadata, list):
                self.daft_df = daft.from_pylist(_metadata)
            else:
                self.daft_df = daft.from_pydict({**_metadata})
            if embeddings is not None:
                self.daft_df = self.add_column("embeddings", embeddings).daft_df
            if not lazy:
                self.daft_df = self.daft_df.collect()

    def __repr__(self) -> str:
        return self.daft_df.__repr__()

    @property
    def on_df(self) -> Wrapper:
        return Wrapper(self)

    @property
    def df(self) -> daft.DataFrame:
        return self.daft_df

    def __len__(self) -> int:
        return self.daft_df.count_rows()

    def __getitem__(self, column: str) -> DaftCollection:
        return self.select(column)

    def cast(
        self, column: str = None, datatype: DataType = DataType.python()
    ) -> DaftCollection:
        if column is None:
            columns = [col(c).cast(datatype) for c in self.column_names]
        else:
            columns = []
            for c in self.column_names:
                if c == column:
                    columns.append(col(column).cast(datatype))
                else:
                    columns.append(c)
        return self.from_daft_df(self.daft_df.select(*columns))

    def add_rows(self, entries: List[Dict[str, Any]]) -> DaftCollection:
        dic = self.to_dict()
        for k in dic:
            for d in entries:
                value = d.get(k, None)
                dic[k].append(value)
        return self.from_data(dic)

    def add(self, entries: List[Dict[str, Any]]) -> DaftCollection:
        return self.add_row(entries)

    def set_embedding_function(self, column: str, embedding_function: Transformation):
        self.embedding_functions[column] = embedding_function

    @property
    def column_names(self) -> List[str]:
        return self.daft_df.column_names

    def from_daft_df(self, df: daft.DataFrame) -> DaftCollection:
        return DaftCollection(
            retriever=self.retriever,
            embedding_functions=self.embedding_functions,
            daft_df=df,
        )

    def from_data(self, data: Any) -> DaftCollection:
        return DaftCollection(
            data=data,
            retriever=self.retriever,
            embedding_functions=self.embedding_functions,
        )

    def collect(self, in_place: bool = False):
        if in_place:
            self.daft_df = self.daft_df.collect(num_preview_rows=None)
            return self
        return self.from_daft_df(self.daft_df.collect(num_preview_rows=None))

    def execute(self) -> DaftCollection:
        return self.collect()

    def to_pandas(self) -> pd.DataFrame:
        collection = self.execute()
        return collection.daft_df.to_pandas()

    def to_dict(self) -> Dict[str, List[Any]]:
        collection = self.execute()
        return collection.daft_df.to_pydict()

    def to_list(self) -> List[Any]:
        collection = self.execute()
        values = list(collection.daft_df.to_pydict().values())
        if len(values) == 1:
            return values[0]
        return values

    def show(self, num_rows: Optional[int] = None):
        if num_rows is None:
            return self.daft_df.show(self.__len__())
        return self.daft_df.show(num_rows)

    @lazy(default=True)
    def iloc(self, idx: Union[int, Iterable[int]]) -> DaftCollection:
        # for some reason this is super slow
        if isinstance(idx, int):
            idx = [idx]

        collection = (
            self.on_df.with_column(
                "_vexpresso_index", indices(col(self.column_names[0]))
            )
            .filter({"_vexpresso_index": {"isin": idx}})
            .exclude("_vexpresso_index")
        )
        return collection

    @lazy(default=True)
    def rename(self, columns: Dict[str, str]) -> DaftCollection:
        expressions = []
        for column in self.column_names:
            if column in columns:
                expressions.append(col(column).alias(columns[column]))
            else:
                expressions.append(col(column))
        return self.on_df.select(*expressions)

    @lazy(default=True)
    def agg(self, *args, **kwargs) -> DaftCollection:
        return self.from_df(self.daft_df.agg(*args, **kwargs))

    @lazy(default=True)
    def add_column(self, name: str, column: List[Any]) -> DaftCollection:
        df = self.df
        if name in self.column_names:
            df = df.exclude(name)
        df = df.with_column("_vexpresso_index", indices(col(self.column_names[0])))
        second_df = daft.from_pydict(
            {name: column, "_vexpresso_index": list(range(len(self)))}
        )

        df = df.join(second_df, on="_vexpresso_index").exclude("_vexpresso_index")
        return self.from_daft_df(df)

    @lazy(default=True)
    def sort(self, column, desc=True) -> DaftCollection:
        return self.from_daft_df(self.daft_df.sort(col(column), desc=desc))

    def embed_query(
        self,
        query: Any,
        embedding_column_name: Optional[str] = None,
        embedding_fn: Optional[Transformation] = None,
        resource_request=ResourceRequest(),
        *args,
        **kwargs,
    ) -> Any:
        return self.embed_queries(
            queries=[query],
            embedding_column_name=embedding_column_name,
            embedding_fn=embedding_fn,
            resource_request=resource_request,
            *args,
            **kwargs,
        )[0]

    def embed_queries(
        self,
        queries: List[Any],
        embedding_column_name: Optional[str] = None,
        embedding_fn: Optional[Transformation] = None,
        resource_request=ResourceRequest(),
        *args,
        **kwargs,
    ) -> Any:
        if embedding_fn is None:
            if embedding_column_name is None:
                raise ValueError("Column name must be provided if embedding_fn is None")
            embedding_fn = self.embedding_functions[embedding_column_name]
        elif isinstance(embedding_fn, str):
            embedding_fn = self.embedding_functions[embedding_fn]

        query_embeddings = (
            daft.from_pydict({"queries": queries})
            .with_column(
                "query_embeddings",
                embedding_fn(col("queries"), *args, **kwargs),
                resource_request=resource_request,
            )
            .select("query_embeddings")
            .collect()
            .to_pydict()["query_embeddings"]
        )
        return query_embeddings

    @lazy(default=True)
    def query(
        self,
        column: str,
        query: List[Any] = None,
        query_embedding: List[Any] = None,
        filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,
        k: int = None,
        sort: bool = True,
        embedding_fn: Optional[Transformation] = None,
        return_scores: bool = False,
        score_column_name: Optional[str] = None,
        resource_request: ResourceRequest = ResourceRequest(),
        retriever: Optional[BaseRetriever] = None,
        *args,
        **kwargs,
    ) -> Collection:
        if query is not None:
            query = [query]

        if query_embedding is not None:
            query_embedding = [query_embedding]

        return self.batch_query(
            column=column,
            queries=query,
            query_embeddings=query_embedding,
            filter_conditions=filter_conditions,
            k=k,
            sort=sort,
            embedding_fn=embedding_fn,
            return_scores=return_scores,
            score_column_name=score_column_name,
            resource_request=resource_request,
            retriever=retriever,
            *args,
            **kwargs,
        )[0]

    @lazy(default=True)
    def batch_query(
        self,
        column: str,
        queries: List[Any] = None,
        query_embeddings: List[Any] = None,
        filter_conditions: Optional[Dict[str, Dict[str, str]]] = None,
        k: int = None,
        sort: bool = True,
        embedding_fn: Optional[Union[Transformation, str]] = None,
        return_scores: bool = False,
        score_column_name: Optional[str] = None,
        resource_request: ResourceRequest = ResourceRequest(),
        retriever: Optional[BaseRetriever] = None,
        *args,
        **kwargs,
    ) -> List[Collection]:
        batch_size = len(queries) if query_embeddings is None else len(query_embeddings)

        if embedding_fn is not None:
            if isinstance(embedding_fn, str):
                embedding_fn = self.embedding_functions[embedding_fn]
            else:
                if column in self.embedding_functions:
                    if embedding_fn != self.embedding_functions[column]:
                        print(
                            "embedding_fn may not be the same as whats in map! Updating what's in map..."
                        )
                self.embedding_functions[column] = get_embedding_fn(embedding_fn)
                embedding_fn = self.embedding_functions[column]

        if query_embeddings is None:
            query_embeddings = self.embed_queries(
                queries,
                column,
                embedding_fn,
                resource_request,
                *args,
                **kwargs,
            )

        if retriever is None:
            retriever = self.retriever

        if k is None:
            k = self.__len__()

        dfs = retrieve(
            batch_size,
            self.daft_df,
            column,
            query_embeddings,
            retriever,
            k,
            sort,
            return_scores,
            score_column_name,
            resource_request,
        )

        for i in range(len(dfs)):
            if filter_conditions is not None:
                dfs[i] = FilterHelper.filter(dfs[i], filter_conditions)

        return [self.from_daft_df(df) for df in dfs]

    @lazy(default=True)
    def select(
        self,
        *args,
    ) -> DaftCollection:
        return self.from_daft_df(FilterHelper.select(self.daft_df, *args))

    @lazy(default=True)
    def exclude(
        self,
        *args,
    ) -> DaftCollection:
        return self.from_daft_df(self.daft_df.exclude(*args))

    @lazy(default=True)
    def filter(
        self, filter_conditions: Dict[str, Dict[str, str]], *args, **kwargs
    ) -> DaftCollection:
        return self.from_daft_df(
            FilterHelper.filter(self.daft_df, filter_conditions, *args, **kwargs)
        )

    @lazy(default=True)
    def apply(
        self,
        transform_fn: Transformation,
        column: DaftCollection,
        *args,
        to: Optional[str] = None,
        resource_request: ResourceRequest = ResourceRequest(),
        datatype: DataType = DataType.python(),
        init_kwargs: Dict[str, Any] = {},
        function: str = "__call__",
        **kwargs,
    ) -> DaftCollection:
        transform_fn = transformation(
            transform_fn, datatype=datatype, init_kwargs=init_kwargs, function=function
        )

        if not isinstance(column, DaftCollection):
            raise TypeError(
                "first args in apply must be a DaftCollection! use `collection['column_name']`"
            )

        collection = self

        args = [column, *args]

        _args = []
        for _arg in args:
            if isinstance(_arg, DaftCollection):
                if len(_arg.column_names) > 1:
                    raise ValueError(
                        "When passing in a Daft collection into `embed`, they must only have 1 column!"
                    )
                column_name = _arg.column_names[0]
                if column_name not in collection.column_names:
                    content = _arg.select(column_name).to_dict()[column_name]
                    collection = collection.add_column(column_name, content)
                _args.append(col(column_name))
            else:
                _args.append(_arg)

        _kwargs = {}
        for k in kwargs:
            _kwargs[k] = kwargs[k]
            if isinstance(_kwargs[k], DaftCollection):
                # only support first column
                column = _kwargs[k].daft_df.columns[0]
                _kwargs[k] = column

        if to is None:
            to = f"tranformed_{_args[0].name()}"

        return collection.on_df.with_column(
            to, transform_fn(*_args, **_kwargs), resource_request=resource_request
        )

    @lazy(default=True)
    def embed(
        self,
        column: Union[DaftCollection, List[Any], str],
        *args,
        embedding_fn: Optional[Transformation] = None,
        to: Optional[str] = None,
        resource_request: ResourceRequest = ResourceRequest(),
        datatype: DataType = DataType.python(),
        init_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> DaftCollection:
        collection = self
        column_name = None
        if isinstance(column, str):
            column_name = column
        elif not isinstance(column, DaftCollection):
            # raw content
            column_name = f"content_{len(collection.column_names)}"
            collection = collection.add_column(column_name, column)
        else:
            column_name = column.column_names[0]

        if to is None:
            to = f"embeddings_{column_name}"

        if embedding_fn is None:
            embedding_fn = self.embedding_functions[to]
        else:
            self.embedding_functions[to] = embedding_fn

        self.embedding_functions[to] = get_embedding_fn(
            self.embedding_functions[to], datatype=datatype, init_kwargs=init_kwargs
        )

        return collection.apply(
            self.embedding_functions[to],
            collection[column_name],
            *args,
            to=to,
            resource_request=resource_request,
            **kwargs,
        )

    def save_local(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        table = self.daft_df.to_arrow()
        pq.write_table(table, os.path.join(directory, "content.parquet"))

    @classmethod
    def from_local_dir(cls, local_dir: str, *args, **kwargs) -> DaftCollection:
        df = daft.read_parquet(os.path.join(local_dir, "content.parquet"))
        return DaftCollection(daft_df=df, *args, **kwargs)

    @classmethod
    def connect(
        cls, address: str = None, cluster_kwargs: Dict[str, Any] = {}, *args, **kwargs
    ) -> DaftCollection:
        import ray

        if address is None:
            addy = ray.init(**cluster_kwargs)
        else:
            addy = ray.init(address=address, **cluster_kwargs)
        daft.context.set_runner_ray(address=addy.address_info["address"])
        return DaftCollection(*args, **kwargs)

    def to_langchain(self, document_column: str, embeddings_column: str):
        from langchain.docstore.document import Document
        from langchain.vectorstores import VectorStore

        class VexpressoVectorStore(VectorStore):
            def __init__(self, collection: DaftCollection):
                self.collection = collection
                self.document_column = document_column
                self.embeddings_column = embeddings_column

            def add_texts(
                self,
                texts: Iterable[str],
                metadatas: Optional[List[dict]] = None,
                **kwargs: Any,
            ) -> List[str]:
                if metadatas is None:
                    metadatas = [{} for _ in range(len(texts))]

                combined = [
                    {self.document_column: t, **m} for t, m in zip(texts, metadatas)
                ]

                self.collection = self.collection.add_rows(combined)

            def similarity_search(
                self, query: str, k: int = 4, **kwargs: Any
            ) -> List[Document]:
                dictionary = self.collection.query(
                    self.embeddings_column, query=query, k=k, lazy=False, **kwargs
                ).to_dict()
                documents = dictionary[self.document_column]
                metadatas = {
                    k: dictionary[k] for k in dictionary if k != self.document_column
                }

                out = []
                for i in range(len(documents)):
                    doc = documents[i]
                    d = {k: metadatas[k][i] for k in metadatas}
                    out.append(Document(page_content=doc, metadata=d))
                return out

            @classmethod
            def from_texts(
                cls,
                *args,
                **kwargs: Any,
            ):
                """Return VectorStore initialized from texts and embeddings."""
                return None

        return VexpressoVectorStore(self)

    @classmethod
    def from_documents(
        cls, documents: List[Document], *args, **kwargs
    ) -> DaftCollection:
        # for langchain integration
        raw = [{"text": d.page_content, **d.metadata} for d in documents]
        return DaftCollection(data=raw, *args, **kwargs)
