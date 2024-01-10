from typing import Any, Optional

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.base import BaseOutputParser
from gpt_index.utils import ErrorToRetry, retry_on_exceptions_with_backoff

import registry
import streaming
from .index_lookup import IndexLookupTool
from index.app_info import QUESTION_KEY, ANSWER_KEY, FOLLOW_UPS_KEY


# TODO: consider few-shot examples
TEMPLATE = '''You are a web3 assistant. You help users use web3 apps, such as Uniswap, AAVE, MakerDao, etc. You assist users in achieving their goals with these protocols, by finding out the information needed to create transactions for users. Your responses should sound natural, helpful, cheerful, and engaging, and you should use easy to understand language with explanations for jargon.

You have access to frequently asked questions about this chat app. Given the knowledge base of questions and answers below for reference, find the best answer to the user's question. If there are any provided follow-up queries from the knowledge base, suggest them to the user in the following format: "Here are some follow-up questions that you could try  asking:", but do not make up your own questions, only use those that are provided.
---
{task_info}
---
User: {question}
Assistant:'''


# how data is retrieved from index
DOCUMENT_TEMPLATE = '''## Knowledge retrieved from index
Question: {question}
Answer: {answer}
Suggested follow-ups: {follow_ups}
'''


@registry.register_class
class IndexAppInfoTool(IndexLookupTool):
    """Tool for searching an app info index and figuring out how to respond to the question."""

    _chain: LLMChain

    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        prompt = PromptTemplate(
            input_variables=["task_info", "question"],
            template=TEMPLATE,
        )
        new_token_handler = kwargs.get('new_token_handler')
        chain = streaming.get_streaming_chain(prompt, new_token_handler)
        super().__init__(
            *args,
            _chain=chain,
            content_description="common questions and answers about the chat assistant app, what it can do, how to interact with it, at a high-level. Only useful for questions about the chat app experience. It does not know specific information about the web3 ecosystem, of tokens or NFTs or contracts, or access to live data and APIs.",
            input_description="a standalone query with all relevant contextual details pertaining to the chat web application",
            output_description="an answer to the question, with suggested follow-up questions if available",
            **kwargs
        )

    def _run(self, query: str) -> str:
        """Query index and answer question using document chunks."""
        docs = retry_on_exceptions_with_backoff(
            lambda: self._index.similarity_search(query, k=self._top_k),
            [ErrorToRetry(TypeError)],
        )
        task_info = '\n'.join([DOCUMENT_TEMPLATE.format(
            question=doc.page_content,
            answer=doc.metadata[ANSWER_KEY],
            follow_ups=doc.metadata[FOLLOW_UPS_KEY],
        ) for doc in docs])
        example = {
            "task_info": task_info,
            "question": query,
            "stop": "User",
        }
        result = self._chain.run(example)
        return result.strip()
