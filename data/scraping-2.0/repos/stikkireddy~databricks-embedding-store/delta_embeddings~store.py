import abc
import time
from dataclasses import dataclass
from typing import Union, Iterator, Tuple, List, Callable, Dict, Any, Optional

import numpy as np
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from delta import *
import functools
import faiss


class DeltaLakeEmbeddingStoreTableInputError(Exception):
    pass


class InvalidTableInputError(Exception):
    pass


@dataclass
class SentenceTransformerModel:
    model: HuggingFaceEmbeddings
    dim: int


@functools.lru_cache(maxsize=None)
def get_st_model(model):
    _embedding_model = HuggingFaceEmbeddings(model_name=model)
    return SentenceTransformerModel(
        _embedding_model
        , len(_embedding_model.embed_query("some_query"))
    )


@F.pandas_udf("array<float>")
def embed(iterator: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
    # needs to be here to avoid pickling issues for udf
    @functools.lru_cache(maxsize=None)
    def get_model(model):
        return HuggingFaceEmbeddings(model_name=model)

    def embed_slow(stack):
        data = stack[0]
        model_name = stack[1]
        model = get_model(model_name)
        return model.embed_documents(data)[0]

    def get_if_one_model(_models: pd.Series):
        models_list: List[str] = np.unique(_models).tolist()
        if len(models_list) == 1:
            return get_model(models_list[0])
        else:
            return None

    def embed_efficient(_data, _models):
        hf_model = get_if_one_model(_models)
        if hf_model is not None:
            return pd.Series(hf_model.embed_documents(_data))
        else:
            stack = np.column_stack((_data, _models))
            return pd.Series(embed_slow(stack))

    for data, models in iterator:
        # Use that state for whole iterator.
        yield embed_efficient(data, models)


# TODO: make this more composable
class FaissIndex(abc.ABC):

    def __init__(self, embedding_model_name):
        self._embedding_model_name: str = embedding_model_name
        self._index: Optional[faiss.Index] = None
        self._raw_id_map: Optional[Dict[int, Any]] = None

    @abc.abstractmethod
    def build(self, **kwargs) -> faiss.Index:
        pass

    @property
    def index(self):
        return self._index

    @abc.abstractmethod
    def search(self, **kwargs) -> np.ndarray:
        pass


# Change to builder
class DefaultEmbeddingsFaissIndex(FaissIndex):
    # Builds a index with the full set of embeddings
    def build(self, **kwargs):
        spark: SparkSession = kwargs["spark"]
        table_name: str = kwargs["table_name"]
        model: SentenceTransformerModel = get_st_model(self._embedding_model_name)
        embeddings_df = spark.table(table_name).filter(
            f"_embeddings_model = '{self._embedding_model_name}' and _embeddings is not null") \
            .select("_id", "raw_id", "_embeddings")
        embeddings_pdf = embeddings_df.toPandas().set_index('_id', drop=False)
        self._raw_id_map = pd.Series(embeddings_pdf.raw_id.values, index=embeddings_pdf._id).to_dict()
        id_index = np.array(embeddings_pdf._id.values).flatten().astype("int")
        embeddings = np.array(embeddings_pdf._embeddings.values.tolist())
        content_encoded_normalized = embeddings.copy()
        faiss.normalize_L2(content_encoded_normalized)
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(model.dim))
        self._index.add_with_ids(content_encoded_normalized, id_index)
        return self._index

    def _search_string(self, query: str, k=10, **kwargs) -> List[List[str]]:
        model: HuggingFaceEmbeddings = get_st_model(self._embedding_model_name).model
        t = time.time()
        query_vector = np.array([np.array(model.embed_query(query), dtype=np.float32)])
        faiss.normalize_L2(query_vector)
        top_k = self._index.search(query_vector, k)
        print('total search time: {} seconds'.format(time.time() - t))
        ids = top_k[1][0].tolist()
        similarities = top_k[0][0].tolist()
        results = [[str(self._raw_id_map[_id]), sim] for _id, sim in zip(ids, similarities)]
        return results

    # def _search_small_df(self, query: Union[str, pd.DataFrame, DataFrame], k=10, **kwargs) -> List[Liststr]:
    #     model: HuggingFaceEmbeddings = get_st_model(self._embedding_model_name).model
    #     t = time.time()
    #     query_vector = np.array([np.array(model.embed_query(query), dtype=np.float32)])
    #     print(query_vector)
    #     print(type(query_vector))
    #     faiss.normalize_L2(query_vector)
    #     top_k = self._index.search(query_vector, k)
    #     print('total search time: {} seconds'.format(time.time() - t))
    #     ids = top_k[1][0].tolist()
    #     # similarities = top_k[0][0].tolist()
    #     results = [str(self._raw_id_map[_id]) for _id in ids]
    #     return results

    # def _search_parallel(self, query: DataFrame, k=10, **kwargs):
    #     use_gpu = kwargs["gpu"]
    #
    #     def search_with_faiss_gpu(index):
    #         def search_content(query_vec, k=15):
    #             query = np.array([query_vec.copy()])
    #             t = time.time()
    #             faiss.normalize_L2(query)
    #             top_k = index.search(query, k)
    #             print('total search time: {} seconds'.format(time.time() - t))
    #             ids = top_k[1][0].tolist()
    #             similarities = top_k[0][0].tolist()
    #             results = embeddings_df.loc[ids]
    #             # results['similarity'] = similarities
    #             return results["raw_id"].tolist()
    #
    #         return search_content
    #
    #     def search_all_items(s: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #         index = index_content
    #         if USE_GPU is True:
    #             res = faiss.StandardGpuResources()  # use a single GPU
    #             index = faiss.index_cpu_to_gpu(res, 0, index_content)
    #         for x in s:
    #             yield x.apply(search_with_faiss_gpu(index))

    def search(self, query: Union[str, pd.DataFrame, DataFrame], k=10, **kwargs) -> List[List[str]]:
        if isinstance(query, str):
            return self._search_string(query, k, **kwargs)

    # query = np.array([query_vec.copy()])
    # t = time.time()
    # faiss.normalize_L2(query)
    # top_k = index.search(query, k)


class DeltaLakeEmbeddingStore:

    def __init__(self,
                 table_name: str,
                 table_location: str = None,
                 spark_session: SparkSession = None):
        # self._embedding_store_table = embedding_store_table
        self._table_name = table_name
        self._table_location = table_location
        self._spark: SparkSession = spark_session

        # loads table dataframe
        self._create_embeddings_table_if_not_exists()
        self._table = self._spark.table(table_name)

    def _create_embeddings_table_if_not_exists(self):
        # TODO: support binary field for images
        id_col_type = "STRING"
        content_col_type = "STRING"
        location_str = "" if self._table_location is None else f"LOCATION '{self._table_location}'"
        print("Creating embedding table if it does not exist...")
        self._spark.sql(f"""
          CREATE TABLE IF NOT EXISTS {self._table_name} (
            _id BIGINT GENERATED ALWAYS AS IDENTITY,
            raw_id {id_col_type} NOT NULL,
            raw_content {content_col_type} NOT NULL,
            raw_metadata MAP<STRING, STRING>,
            _embeddings_model STRING NOT NULL,
            _embeddings ARRAY<FLOAT>,
            _sha256 STRING GENERATED ALWAYS AS (cast(sha2(raw_content, 256) as STRING))
          ) USING DELTA
          {location_str}
          ;
        """)

    def _get_normalized_df(self, input_data: Union[pd.DataFrame, DataFrame]) -> DataFrame:
        if len(input_data.columns) not in [2, 3]:
            raise InvalidTableInputError(
                "Input data should contain atleast 2 columns, id, content and optionally metadata"
            )
        spark_df = input_data if isinstance(input_data, DataFrame) else self._spark.createDataFrame(input_data)
        if len(spark_df.columns) == 2:
            spark_df = spark_df.withColumn("metadata", F.lit(None))
        return spark_df

    def _create_entries(self, df: DataFrame, sentence_transformer_model_name: str):
        table = DeltaTable.forName(self._spark, self._table_name)
        id_col_name = df.columns[0]
        content_col_name = df.columns[1]
        metadata_col_name = df.columns[2]
        table.alias("tgt") \
            .merge(df
                   .withColumn("_sha256", F.sha2(content_col_name, 256))
                   .withColumn("_embeddings_model", F.lit(sentence_transformer_model_name))
                   .alias('src'),
                   f"src.{id_col_name} = tgt.raw_id and tgt._embeddings_model = src._embeddings_model") \
            .whenMatchedUpdate(
            condition="tgt._sha256 <> src._sha256",
            set={
                "raw_content": f"src.{content_col_name}",
                "_embeddings": "null"
            }
        ) \
            .whenNotMatchedInsert(
            values={
                "raw_id": f"src.{id_col_name}",
                "raw_content": f"src.{content_col_name}",
                "raw_metadata": f"src.{metadata_col_name}",
                "_embeddings_model": "src._embeddings_model",
            }
        ) \
            .execute()

    def _get_embedding_candidates(self, input_data: DataFrame,
                                  sentence_transformer_model_name):
        tgt = self._table
        id_col_name = input_data.columns[0]
        return input_data.alias("src").join(tgt.alias("tgt")
                                            .filter(f"_embeddings_model = '{sentence_transformer_model_name}'"),
                                            input_data[id_col_name] == tgt.raw_id, "inner") \
            .select(f"tgt.*").filter("_embeddings is null")

    def _apply_embeddings(self,
                          input_data: DataFrame,
                          sentence_transformer_model_name):
        table = DeltaTable.forName(self._spark, self._table_name)
        df = self._get_embedding_candidates(input_data, sentence_transformer_model_name)
        print(f"Processing embeddings: {df.count()}")
        df = df.withColumn("_computed_embeddings", embed("raw_content", "_embeddings_model"))
        table.alias("tgt") \
            .merge(df.alias('src'),
                   f"src._id = tgt._id and src.raw_id = tgt.raw_id and tgt._embeddings_model = src._embeddings_model") \
            .whenMatchedUpdate(
            set={
                "_embeddings": "src._computed_embeddings"
            }
        ) \
            .execute()

    # no tokenizers/preprocessing
    # first column should be id, second text and last metadata optional
    def add_and_embed_documents(self,
                                input_data: Union[pd.DataFrame, DataFrame],
                                sentence_transformer_model_name):
        print("Getting normalized df...")
        norm_df = self._get_normalized_df(input_data)
        print("Creating entries...")
        self._create_entries(norm_df, sentence_transformer_model_name)
        print("Processing embeddings...")
        self._apply_embeddings(norm_df, sentence_transformer_model_name)

    def self_sim_search(self, sentence_transformers_model_name,
                        use_gpu=False,
                        translate_ids=True,
                        custom_index=None,
                        num_partitions=None,
                        **kwargs):
        # TODO: allow user to provide search space
        # Optional KWARGS: use_gpu, translate_ids can cause broadcast issues if too many ids,
        # faiis index only accepts 64 bit ints
        filtered_df = (self._table
                       .filter(f"_embeddings_model = '{sentence_transformers_model_name}'")
                       .filter(f"_embeddings is not null")
                       )
        if num_partitions is not None:
            print(f"Configuring dataframe to have {num_partitions} partitions")
            filtered_df = filtered_df.repartition(num_partitions).cache()
            filtered_df.count()

        raw_id_lookup_map = None
        if translate_ids is True:
            print("Generating translation ids for faiss index lookup... "
                  "may take a while or fail broadcasting if too many vectors")
            print("To disable please set translate_ids to False")
            embeddings = filtered_df.toPandas()
            raw_id_lookup_map = dict(zip(embeddings._id, embeddings.raw_id))
            print(f"Finished fetching ids... {len(raw_id_lookup_map.keys())} keys being used")

        if custom_index is None:
            idx = DefaultEmbeddingsFaissIndex(sentence_transformers_model_name)
            idx.build(spark=self._spark, table_name=self._table_name)
        else:
            idx = custom_index

        def search_with_faiss_efficient(index, id_lookup_map=None):
            def search_content(query_vec, k=15):
                query = np.array(query_vec.tolist())
                t = time.time()
                faiss.normalize_L2(query)
                top_k = index.search(query, k)
                # logging in executor
                print('total search time: {} seconds'.format(time.time() - t))
                ids = top_k[1].tolist()
                similarities = top_k[0].tolist()
                if id_lookup_map is None:
                    return ids, similarities
                return [np.vectorize(id_lookup_map.get)(id_np_arr) for id_np_arr in ids], similarities

            return search_content

        search_index = idx.index

        def search_all_items(s: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
            index = search_index
            if use_gpu is True:
                res = faiss.StandardGpuResources()  # use a single GPU
                index = faiss.index_cpu_to_gpu(res, 0, search_index)
            for x in s:
                ids, scores = search_with_faiss_efficient(index, raw_id_lookup_map)(x)
                id_series = pd.Series(ids)
                scores_series = pd.Series(scores)
                yield pd.concat([id_series, scores_series], axis=1)

        if raw_id_lookup_map is not None:
            search_all_items_pudf = F.pandas_udf("struct<ids:array<string>,scores:array<float>>")(search_all_items)
        else:
            search_all_items_pudf = F.pandas_udf("struct<ids:array<int>,scores:array<float>>")(search_all_items)

        return filtered_df \
            .withColumn("sim_search", search_all_items_pudf(F.col('_embeddings'))) \
            .selectExpr("*", "sim_search['ids'] as similar_ids", "sim_search['scores'] as similarity_scores")

    # # no tokenizers/preprocessing
    # # softmax
    # def search(self,
    #            df,
    #            sentence_transformers_model_name,
    #            faiss_index_class,
    #            index_config=None,
    #            search_space_embeddings_func: Callable = _get_embeddings_default,
    #            faiss_index_func: Callable = _get_embeddings_default,
    #            k=15):
    #     embeddings_df = search_space_embeddings_func(self._spark, self._table_name, sentence_transformers_model_name)
    #
    # def search_and_save(self, df, faiss_index_class, index_config=None):
    #     ...
