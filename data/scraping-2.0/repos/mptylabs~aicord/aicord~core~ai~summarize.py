from __future__ import annotations

import logging
import os

from langchain.schema import (
    Document,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_loaders import base as chat_loaders
from langchain.chat_loaders.utils import (
    map_ai_messages,
    merge_chat_runs,
)

# Set up logging
logger = logging.getLogger(__name__)

question_template = \
"""
You are an expert in summarizing conversations in public text chats.
Your goal is to create a summary of the conversation and extract main topics that were discussed, \
who started topic and who participated in this topic conversation.
Below you can find the transcript of the conversation:
--------
{text}
--------

SUMMARY AND TOPIC BULLET LIST WITH THEIR AUTHORS AND PARTICIPANTS:
"""
question_prompt = PromptTemplate.from_template(question_template)

openai_api_key = os.environ['OPENAI_TOKEN']
model_name = "gpt-4-1106-preview"
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name=model_name)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    model_name=model_name,
    # chunk_size=1000,
    # chunk_overlap=200,
)
chain = load_summarize_chain(
    llm,
    chain_type="stuff",
    prompt=question_prompt,
    # chain_type="refine",
    # question_prompt=question_prompt,
    # refine_prompt=refine_prompt,
    verbose=True,
)
