# Copyright 2023 Qarik Group, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, List

from common import admin_dao, constants, gcs_tools, solution
from common.cache import cache
from common.log import Logger, log, log_params
from langchain.llms.openai import OpenAIChat
from llama_index import (Document, GPTSimpleKeywordTableIndex, GPTVectorStoreIndex, LLMPredictor, ServiceContext,
                         SimpleDirectoryReader, StorageContext, load_index_from_storage)
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
# import google.generativeai as palm
# from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.query_engine.transform_query_engine import TransformQueryEngine

# from llama_index.selectors.llm_selectors import LLMSingleSelector
# from llama_index.tools.query_engine import QueryEngineTool

logger = Logger(__name__).get_logger()
logger.info('Initializing...')

DATA_LOAD_LOCK = threading.Lock()
"""Block many concurrent data loads at once."""

LLAMA_FILE_LOCK = threading.Lock()
"""Lock to prevent concurrent updates of the same index - needed in case we have more than one request processing."""

INDEX_BUCKET: str = solution.getenv('EMBEDDINGS_BUCKET_NAME')
"""Location to download llama-index embeddings from."""

LAST_LOCAL_INDEX_UPDATE: datetime | None = None
"""Keep track of the most recent local index update to avoid unnecessary refreshes."""

if solution.LOCAL_DEVELOPMENT_MODE:
    LLAMA_INDEX_DIR: str = 'dev/tmp/llamaindex-embeddings'
else:
    LLAMA_INDEX_DIR = 'tmp/llamaindex-embeddings'

LOCAL_DEV_DATA_DIR: str = 'dev/tmp'
"""Location of the local data directory for development on local machine."""


@log
def _get_llm(provider: constants.LlmProvider) -> LLMPredictor:
    """Return LLM predictor."""
    if provider == constants.LlmProvider.OPEN_AI:
        llm = LLMPredictor(llm=OpenAIChat(temperature=constants.TEMPERATURE, model_name=constants.GPT_MODEL))  # type: ignore
    else:
        raise ValueError(f'Unknown LLM provider: {provider}')
    return llm


@log_params
def load_resumes(resume_dir: str | None) -> dict[str, List[Document]]:
    """Initialize list of resumes from index storage or from the directory with PDF source files."""
    resumes: dict[str, List[Document]] = {}
    if resume_dir is None:
        resume_dir = ''
    resume_path = Path(resume_dir)
    index_path = Path(LLAMA_INDEX_DIR)
    global DATA_LOAD_LOCK
    with DATA_LOAD_LOCK:
        if index_path.exists():
            logger.info('Loading people names (not resumes) from existing index storage...')
            names = glob.glob(f'{index_path}/*',)

            if len(names):
                for file_name in names:
                    # We do not care about the contents of the resume because it will be loaded from index
                    # All we care for here is the name - aka the Key, not Value
                    resumes[Path(file_name).name] = []
                return resumes
            else:
                logger.warning('No resumes found in the index directory: %s', index_path)
                logger.warning('Removing the index storage directory: %s', index_path)
                Path.rmdir(index_path)

        logger.info('Loading people names from the source dir with resume PDF files...')
        Path.mkdir(resume_path, parents=True, exist_ok=True)

        # Check if there are any pdf files in the data directory
        pdf_files = glob.glob(f'{resume_path}/*.pdf')

        if len(pdf_files):
            # Each resume shall be named as '<person_name>.pdf' optionally with 'resume' suffix
            for resume in pdf_files:
                person_name = os.path.basename(resume).replace('.pdf', '').replace(
                    'Resume', '').replace('resume', '').replace('_', ' ').strip()
                logger.debug(f'Loading: {person_name}')
                resume_content = SimpleDirectoryReader(input_files=[resume]).load_data()
                resumes[person_name] = resume_content
        else:
            logger.warning('No resume PDF files found in the data directory: %s', resume_path)

    return resumes


@log
def _load_resume_indices(resumes: dict[str, List[Document]],
                        service_context: ServiceContext, embeddings_dir: str) -> dict[str, GPTVectorStoreIndex]:
    """Load or create index storage contexts for each person in the resumes list."""
    vector_indices = {}
    for person_name, resume_data in resumes.items():
        cache_file_path = Path(f'./{embeddings_dir}/{person_name}')
        if cache_file_path.exists():
            logger.debug('Loading index from storage file: %s', cache_file_path)
            storage_context = StorageContext.from_defaults(persist_dir=str(cache_file_path))
            vector_indices[person_name] = load_index_from_storage(storage_context=storage_context)
        else:
            storage_context = StorageContext.from_defaults()
            # build vector index
            vector_indices[person_name] = GPTVectorStoreIndex.from_documents(
                resume_data,
                service_context=service_context,
                storage_context=storage_context,
            )
            # set id for vector index
            # vector_indices[person_name].index_struct.index_id = person_name
            vector_indices[person_name].set_index_id(person_name)
            logger.debug('Saving index to storage file: %s', cache_file_path)
            storage_context.persist(persist_dir=str(cache_file_path))

    # ------------------- Test
    # name = 'Roman Kharkovski'
    # test_query = f'What are the main skills for {name}?'
    # logger.debug('Test query: %s', test_query)
    # response = vector_indices[f'{name}'].as_query_engine().query(test_query)
    # logger.debug('Response: %s', str(response))
    # exit(0)
    # ------------------- end of test

    return vector_indices    # type: ignore


@log
def _load_resume_index_summary(resumes: dict[str, Any]) -> dict[str, str]:
    index_summaries = {}
    for person_name in resumes.keys():
        # index_summaries[person_name] = (f'Use this index if you need to lookup specific facts about {person_name}.')
        index_summaries[person_name] = (f'This content contains resume of {person_name}.\n'
                                        f'Use this index if you need to lookup specific facts about {person_name}.\n'
                                        'Do not confuse people with the same lastname, but different first names.'
                                        'If you cant find the answer, respond with the best of your knowledge.'
                                        'Do not use this index if you want to analyze multiple people.')
    return index_summaries


@log_params
def generate_embeddings(resume_dir: str, provider: constants.LlmProvider) -> None:
    """Generate embeddings from PDF resumes."""
    resumes = load_resumes(resume_dir=resume_dir)
    if not resumes:
        return None
    predictor = _get_llm(provider=provider)
    context = ServiceContext.from_defaults(llm_predictor=predictor, chunk_size_limit=constants.CHUNK_SIZE)
    _load_resume_indices(resumes=resumes, service_context=context, embeddings_dir=LLAMA_INDEX_DIR)


@log_params
def _get_resume_query_engine(provider: constants.LlmProvider, resume_dir: str | None = None) -> BaseQueryEngine | None:
    """Load the index from disk, or build it if it doesn't exist."""
    llm = _get_llm(provider=provider)
    service_context = ServiceContext.from_defaults(llm_predictor=llm, chunk_size_limit=constants.CHUNK_SIZE)

    resumes: dict[str, List[Document]] = load_resumes(resume_dir=resume_dir)
    logger.debug('-------------------------- resumes: %s', resumes.keys())
    if not resumes:
        return None
    # vector_indices = load_resume_indices(resumes, service_context)
    vector_indices = _load_resume_indices(resumes=resumes, service_context=service_context,
                                         embeddings_dir=LLAMA_INDEX_DIR)
    index_summaries = _load_resume_index_summary(resumes)

    graph = ComposableGraph.from_indices(root_index_cls=GPTSimpleKeywordTableIndex,
                                         children_indices=[index for _, index in vector_indices.items()],
                                         index_summaries=[summary for _, summary in index_summaries.items()],
                                         max_keywords_per_chunk=constants.MAX_KEYWORDS_PER_CHUNK)

    # root_index = graph.get_index(graph.root_id)
    root_index = graph.get_index(index_struct_id=graph.root_id)
    root_index.set_index_id('compare_contrast')
    graph.index_struct.summary = ('This index contains resumes of multiple people. '
                                  'Do not confuse people with the same lastname, but different first names.'
                                  'Use this index if you want to compare multiple people.')

    decompose_transform = DecomposeQueryTransform(llm, verbose=True)

    custom_query_engines = {}
    for index in vector_indices.values():
        query_engine = index.as_query_engine(service_context=service_context,
                                             similarity_top_k=constants.SIMILARITY_TOP_K)
        query_engine = TransformQueryEngine(query_engine=query_engine,
                                            query_transform=decompose_transform,
                                            transform_metadata={'index_summary': index.index_struct.summary},
                                            )  # type: ignore
        custom_query_engines[index.index_id] = query_engine

    custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
        retriever_mode='simple',
        response_mode='tree_summarize',
        service_context=service_context,
        verbose=True,
        use_async=True,
    )

    graph_query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)

    # ------------------- Test
    # name1 = 'Roman Kharkovski'
    # name2 = 'Steven Kim'
    # response = graph_query_engine.query(f'Compare and contrast the skills of {name1} and {name2}.')
    # logger.debug('Response: %s', str(response))
    # ------------------- end of test

    return graph_query_engine

    # TODO: the query engine tool does not longer work - need to debug
    # query_engine_tools = []

    # # add vector index tools
    # for person_name in resumes.keys():
    #     index = vector_indices[person_name]
    #     summary = index_summaries[person_name]
    #     query_engine = index.as_query_engine(service_context=service_context)
    #     vector_tool = QueryEngineTool.from_defaults(query_engine=query_engine, description=summary)
    #     query_engine_tools.append(vector_tool)

    # # add graph tool
    # graph_tool = QueryEngineTool.from_defaults(graph_query_engine, description=graph.index_struct.summary)
    # query_engine_tools.append(graph_tool)
    # router_query_engine = RouterQueryEngine.from_defaults(selector=LLMSingleSelector.from_defaults(
    #     service_context=service_context), query_engine_tools=query_engine_tools)

    # return router_query_engine


@cache
@log
def _refresh_llama_index() -> None:
    """Refresh the index of resumes from the database using Llama-Index."""
    global LAST_LOCAL_INDEX_UPDATE

    if solution.LOCAL_DEVELOPMENT_MODE:
        logger.info('Running in local development mode')
        index_path = Path(LLAMA_INDEX_DIR)
        if not index_path.exists():
            # TODO - need to generate proper embeddings for each provider, not hard coded
            generate_embeddings(resume_dir=LOCAL_DEV_DATA_DIR, provider=constants.LlmProvider.OPEN_AI)
        return

    global LLAMA_FILE_LOCK
    last_resume_refresh = admin_dao.AdminDAO().get_resumes_timestamp()
    if LAST_LOCAL_INDEX_UPDATE is None or LAST_LOCAL_INDEX_UPDATE < last_resume_refresh:
        logger.info('Refreshing local index of resumes...')
        # Prevent concurrent updates of the same index - needed in case we have more than one request processing
        with LLAMA_FILE_LOCK:
            # Check for condition again because the index may have been updated while we were waiting for the lock
            if LAST_LOCAL_INDEX_UPDATE is None or LAST_LOCAL_INDEX_UPDATE < last_resume_refresh:
                gcs_tools.download(bucket_name=INDEX_BUCKET, local_dir=LLAMA_INDEX_DIR)
        return last_resume_refresh

    logger.info('Skipping refresh of resumes index because no changes in source resumes were detected.')
    LAST_LOCAL_INDEX_UPDATE = last_resume_refresh


@log
def query(question: str) -> str:
    """Run LLM query for CHatGPT."""
    _refresh_llama_index()
    query_engine = _get_resume_query_engine(provider=constants.LlmProvider.OPEN_AI)
    if query_engine is None:
        raise SystemError('No resumes found in the database. Please upload resumes.')

    return str(query_engine.query(question))
