# Copyright 2023 Google LLC
# Copyright 2023 Qarik Group, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Query methods for VertexAI Vector Search.

This code is based on the following Google Cloud Platform example:
https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/document-qa/question_answering_documents_langchain_matching_engine.ipynb
"""

import textwrap

from common import solution
from common.log import Logger, log
# import vertexai
from google.cloud import aiplatform
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI  # type: ignore
from langchain.prompts import PromptTemplate
from matching_engine import CustomVertexAIEmbeddings, MatchingEngine
from matching_engine_tools import MatchingEngineUtils

logger = Logger(__name__).get_logger()
logger.info('Initializing...')


PROJECT_ID: str = solution.PROJECT_ID
"""GCP Project ID for this project."""
REGION: str = solution.REGION
"""Vertex AI region."""
ME_REGION: str = solution.REGION
"""Matching engine region."""
ME_INDEX_NAME: str = f'{PROJECT_ID}-me-index'
"""Matching engine index name."""
SOURCE_PDF_BUCKET: str = solution.getenv('RESUME_BUCKET_NAME')
"""GCS bucket where source PDF files are stored."""
ME_EMBEDDING_BUCKET: str = f'matching-engine-embeddings-{PROJECT_ID}'
"""GCS bucket where Matching Engine index and embeddings are stored."""
EMBEDDING_QPM: int = 100
"""Rate limit for calling Google VertexAI embeddings API."""
TEMPERATURE: float = 0.0
"""Temperature for LLM."""
TOP_P: float = 1.0
"""Top P for LLM."""
TOP_K: int = 40
"""Top K for LLM."""
NUMBER_OF_RESULTS = 20
"""Number of results to return from the Matching Engine."""
SEARCH_DISTANCE_THRESHOLD = 0.6
"""Search distance threshold for the Matching Engine."""


logger.debug('Vertex AI SDK version: %s', aiplatform.__version__)


def _formatter(result):
    """Utility function to format the result."""
    print(f'Query: {result["query"]}')
    print('.' * 80)
    if 'source_documents' in result.keys():
        for idx, ref in enumerate(result['source_documents']):
            print('-' * 80)
            print(f'REFERENCE #{idx}')
            print('-' * 80)
            if 'score' in ref.metadata:
                print(f'Matching Score: {ref.metadata["score"]}')
            if 'source' in ref.metadata:
                print(f'Document Source: {ref.metadata["source"]}')
            if 'document_name' in ref.metadata:
                print(f'Document Name: {ref.metadata["document_name"]}')
            print('.' * 80)
            print(f'Content: \n{_wrap(ref.page_content)}')
    print('.' * 80)
    print(f'Response: {_wrap(result["result"])}')
    print('.' * 80)


def _wrap(s):
    return '\n'.join(textwrap.wrap(s, width=120, break_long_words=False))


# vertexai.init(project=PROJECT_ID, location=REGION)

logger.debug('Initialize VertexAI LangChain Models...')
_llm = VertexAI(
    model_name='text-bison@001',
    max_output_tokens=1024,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    verbose=True,
)

logger.debug('Creating custom embeddings class...')
_embeddings = CustomVertexAIEmbeddings(requests_per_minute=EMBEDDING_QPM)

logger.debug('Creating matching engine utils...')
_mengine = MatchingEngineUtils(project_id=PROJECT_ID, region=ME_REGION, index_name=ME_INDEX_NAME)
ME_INDEX_ID, ME_INDEX_ENDPOINT_ID = _mengine.get_index_and_endpoint()

# Initialize Matching Engine vector store with text embeddings model
_me = MatchingEngine.from_components(
    project_id=PROJECT_ID,
    region=ME_REGION,
    gcs_bucket_name=f'gs://{ME_EMBEDDING_BUCKET}'.split('/')[2],
    embedding=_embeddings,
    index_id=ME_INDEX_ID,
    endpoint_id=ME_INDEX_ENDPOINT_ID,
)


"""
LangChain provides easy ways to chain multiple tasks that can do QA over a set of documents, called QA chains. The notebook works with [**RetrievalQA**](https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa.html) chain which is based on **load_qa_chain** under the hood.
"""

# Expose index to the retriever
_retriever = _me.as_retriever(
    search_type='similarity',
    search_kwargs={
        'k': NUMBER_OF_RESULTS,
        'search_distance': SEARCH_DISTANCE_THRESHOLD,
    },
)

# Customize the default retrieval prompt template
template = """SYSTEM: You are an intelligent assistant answering questions about people and their skills from their resumes.
Use the following pieces of context to answer the question at the end. Do not try to make up an answer.
=============
{context}
=============
Question: {question}
Helpful Answer:"""

# Configure RetrievalQA chain. Uses LLM to synthesize results from the search index.
_qa = RetrievalQA.from_chain_type(
    llm=_llm,
    chain_type='stuff',
    retriever=_retriever,
    return_source_documents=True,
    verbose=True,
    chain_type_kwargs={
        'prompt': PromptTemplate(
            template=template,
            input_variables=['context', 'question'],
        ),
    },
)

# Enable verbose logging for debugging and troubleshooting the chains which includes the complete prompt to the LLM
_qa.combine_documents_chain.verbose = True
_qa.combine_documents_chain.llm_chain.verbose = True  # type: ignore
_qa.combine_documents_chain.llm_chain.llm.verbose = True  # type: ignore

@log
def query(question: str, qa=_qa, k=NUMBER_OF_RESULTS, search_distance=SEARCH_DISTANCE_THRESHOLD) -> str:
    """Ask a question to the Vertex PaLM model. This is main exposed method of this module."""
    qa.retriever.search_kwargs['search_distance'] = search_distance
    qa.retriever.search_kwargs['k'] = k
    result = qa({'query': question})
    _formatter(result)
    return str(result['result'])
