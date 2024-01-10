from typing import Any, Optional

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.base import BaseOutputParser

import registry
import streaming
from .index_lookup import IndexLookupTool
from gpt_index.utils import ErrorToRetry, retry_on_exceptions_with_backoff
import utils.timing as timing


TEMPLATE = '''**Immediate Action & Review Needed**: Every time you mention specific platforms, tools, technologies, or any topic deserving of a URL, you **must** incorporate it into the text using markdown-style linking. There are two ways to do this:

1. Seamlessly embed the URL into descriptive text.
2. If you need to specify the exact URL for clarity, make sure it is still formatted in markdown.

Here's your blueprint:

**Correct - Embedded**: Learn more about [Ethereum](https://www.ethereum.org/).
**Correct - Explicit**: Visit the Ethereum website at [https://www.ethereum.org/](https://www.ethereum.org/).
**Incorrect**: Learn more at https://www.ethereum.org/ or "Visit the Ethereum website here: https://www.ethereum.org/".

Being a web3 assistant, aim to deliver answers that are clear, engaging, and most importantly, user-friendly. Web3 topics can be intricate, so your goal is to be the bridge to understanding. Always simplify jargon and ensure URLs are user-friendly and clickable.

Before finalizing any response, stop and verify: "Did I format all URLs in markdown?"

If you can't provide an answer, it's perfectly fine to admit it. But regardless of the content of your response, ensure all URLs are **formatted correctly**.
---
{task_info}
---

User: {question}
Assistant:'''


@registry.register_class
class IndexLinkSuggestionTool(IndexLookupTool):
    """Tool for searching a document index and summarizing results to answer the question."""

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
            output_description="a summarized answer with source citations",
            **kwargs
        )

    def _run(self, query: str) -> str:
        """Query index and answer question using document chunks."""

        docs = retry_on_exceptions_with_backoff(
            lambda: self._index.similarity_search(query, k=self._top_k),
            [ErrorToRetry(TypeError)],
        )
        timing.log('widget_index_lookup_done')

        task_info = ""
        for i, doc in enumerate(docs):
            dapp_info = f"""### DAPP {i+1}\nname: {doc.metadata['name']}\ndescription: {doc.page_content}\nurl: {doc.metadata['url']}\n\n"""
            task_info += dapp_info

        example = {
            "task_info": task_info,
            "question": query,
            "stop": "User",
        }
        self._chain.verbose = True
        result = self._chain.run(example)

        return result.strip()

   


