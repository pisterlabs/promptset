from typing import List

import yaml
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.callbacks.base import AsyncCallbackManager

from src.callback import StreamingLLMCallbackHandler
from src.prompts import (
    SOURCES_PROMPT_TEMPLATE,
    FILTER_DOCUMENTS_PROMPT_TEMPLATE,
)


def build_yaml_documents(query_results: List) -> str:
    yaml_doc_template = "```yaml\n{document}\n```"
    documents = [{
        "id": idx + 1,
        "content": doc.page_content,
        "source": doc.metadata["source"]
    } for idx, doc in enumerate(query_results)]
    yaml_documents = [
        yaml_doc_template.format(document=yaml.safe_dump(info))
        for info in documents
    ]
    return "\n".join(yaml_documents)


def get_filter_documents_chain(api_key: str) -> LLMChain:
    llm = OpenAI(
        openai_api_key=api_key,
        streaming=False,
        verbose=True,
        temperature=0.0,
        max_tokens=1000,
    )
    prompt = PromptTemplate(
        template=FILTER_DOCUMENTS_PROMPT_TEMPLATE,
        input_variables=["question", "documents"]
    )
    return LLMChain(llm=llm, prompt=prompt)


def get_streaming_chain(stream_handler: StreamingLLMCallbackHandler, api_key: str) -> LLMChain:
    stream_manager = AsyncCallbackManager([stream_handler])
    llm = OpenAI(
        openai_api_key=api_key,
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0.0,
        max_tokens=500,
    )
    template = SOURCES_PROMPT_TEMPLATE
    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "documents"]
    )
    manager = AsyncCallbackManager([])
    return LLMChain(llm=llm, prompt=prompt, callback_manager=manager)
