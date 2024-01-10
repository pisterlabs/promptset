# Copyright 2023 Qarik Group, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License');
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
"""Main service that handles REST API calls with user questions and invokes backend LLMs to get responses."""

from typing import Annotated, Any

import chat_dao
import langchain_tools
from common import api_tools, constants, llamaindex_tools, solution
from common.log import Logger, log_params
from fastapi import Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query_engine import goog_search_tools, vertexai_tools
from query_engine.chat_dao import VoteStatistic

logger = Logger(__name__).get_logger()
logger.info('Initializing...')


class AskInput(BaseModel):
    """Input parameters for the ask endpoint."""
    question: str
    """Question to ask the LLM model."""
    prompt_prefix: str = ''
    """Prefix to add to the question before passing it to the LLM model."""


class VoteInput(BaseModel):
    """Input parameters for the vote endpoint."""
    llm_backend: str
    question: str
    answer: str
    upvoted: bool


app = api_tools.ServiceAPI(title='Resume Chatbot API (experimental)',
                           description='Request / response API for the Resume Chatbot that uses LLM for queries.')

# In case you need to print the log of all inbound HTTP headers
# app.router.route_class = api_tools.DebugHeaders

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

_users_db = chat_dao.UserDao()
"""Data Access Object to the database of users."""

_vote_db = chat_dao.VoteDao()
"""Data Access Object to the database of votes."""


def _get_user_id(user_email: str | None) -> str:
    """Extract user ID from the request headers.˝"""
    user_id: str = 'anonymous'
    if user_email is None:
        logger.warning('No authenticated user email found in the request headers.')
    else:
        # Header passed from IAP
        # Extract part of the x_goog_authenticated_user_email string after the ':'
        # Example: accounts.google.com:rkharkovski@qarik.com -> rkharkovski@qarik.com
        user_id = user_email.split(':')[-1]
    logger.debug('Received question from user: %s', user_id)
    return user_id


def _store_answer(data: AskInput, answer: str, x_goog: Any, provider: constants.LlmProvider):
    _users_db.save_question_answer(user_id=_get_user_id(x_goog),
                                   question=f'{data.prompt_prefix}===>{data.question}',
                                   answer=str(answer),
                                   llm_backend=str(provider))


@app.get('/people')
@log_params
def list_people() -> list[str]:
    """List all people names found in the database of uploaded resumes."""
    llamaindex_tools._refresh_llama_index()
    people = llamaindex_tools.load_resumes(resume_dir='')
    return [person for person in people.keys()]


@app.get('/health', name='Health check and information about the software version and configuration.')
@log_params
def healthcheck() -> dict:
    """Verify that the process is up without testing backend connections."""
    return solution.health_status()


@app.post('/ask_gpt', name='Ask a question to the GPT-3 model using LlamaIndex and local embeddings store.'
          ' This can be slow because of LlamaIndex chain implementation.')
@log_params
def ask_gpt(data: AskInput, x_goog_authenticated_user_email: Annotated[str | None, Header()] = None) -> dict[str, str]:
    """Ask a question to the GPT-3 model."""
    answer = llamaindex_tools.query(question=f'{data.prompt_prefix}\n{data.question}')
    _store_answer(data=data,
                  answer=answer,
                  x_goog=x_goog_authenticated_user_email,
                  provider=constants.LlmProvider.OPEN_AI)
    return {'answer': str(answer)}


@app.post('/ask_ent_search', name='Ask a question to the Google GenAI using Enterprise Search with summarization.')
@log_params
def ask_goog_ent_search(data: AskInput, x_goog_authenticated_user_email: Annotated[str | None, Header()] = None) -> dict[str, str]:
    """Ask a question to the Google GenAI using Enterprise Search with summarization."""
    answer = goog_search_tools.query(question=f'{data.prompt_prefix}\n{data.question}')
    _store_answer(data=data,
                  answer=answer,
                  x_goog=x_goog_authenticated_user_email,
                  provider=constants.LlmProvider.GOOG_ENT_SEARCH)
    return {'answer': str(answer)}


@app.post('/ask_palm_chroma_langchain',
          name='Ask a question to the Google PaLM model using local index store in ChromaDB and Langchain.')
@log_params
def ask_palm_chroma_langchain(data: AskInput,
                              x_goog_authenticated_user_email: Annotated[str | None, Header()] = None) -> dict[str, str]:
    """Ask a question to the Google PaLM model using local index store in ChromaDB and Langchain."""
    answer = langchain_tools.query(question=f'{data.prompt_prefix}\n{data.question}')
    _store_answer(data=data,
                  answer=answer,
                  x_goog=x_goog_authenticated_user_email,
                  provider=constants.LlmProvider.GOOG_PALM)
    return {'answer': answer}


@app.post('/ask_vertexai',
          name='Ask a question to the Google PaLM 2 model via Langchain using VertexAI Embeddings and Index Search.')
@log_params
def ask_vertexai(data: AskInput, x_goog_authenticated_user_email: Annotated[str | None, Header()] = None) -> dict[str, str]:
    """Ask a question to the Google PaLM 2 model via Langchain using VertexAI Embeddings and Index Search.

    This should scale well for large datasets."""
    answer = vertexai_tools.query(data.question)
    _store_answer(data=data,
                  answer=answer,
                  x_goog=x_goog_authenticated_user_email,
                  provider=constants.LlmProvider.GOOG_VERTEX)
    return {'answer': answer}


@app.post('/vote', name='Submit user vote for the LLM answer. Returns total number of votes for all LLMs.')
@log_params
def vote(data: VoteInput) -> list[VoteStatistic]:
    """Submit user vote for the provided response.

    Returns: total number of votes for each registered LLM backend. For example:
    [{'name': 'ChatGPT',
        'up': 11,
        'down': -22, },
    {'name': 'Google Enterprise Search',
        'up': 33,
        'down': -44, },
    {'name': 'Google VertexAI',
        'up': 77,
        'down': -88, },
    ]
    """
    _vote_db.submit_vote(llm=data.llm_backend,
                         question=data.question,
                         answer=data.answer,
                         upvoted=data.upvoted)
    return _vote_db.get_llm_totals()
