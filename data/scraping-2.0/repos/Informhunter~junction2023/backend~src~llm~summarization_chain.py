import re

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda

from src.llm.openai import OPENAI_SUMMARIZATION_MODEL


_SUMMARIZATION_PROMPT_TEMPLATE = """
Write a concise summary of the following text. Summary should contain only top 3 the most important statements.
Each statement should be no longer than 10 words. You **should** always prefer short summaries.
Do NOT include title in the summary.

Output format:
```<statement 1> <statement 2> <statement 3>```

```{text}```
CONCISE SUMMARY:
""".strip()


_SUMMARIZATION_PROMPT = PromptTemplate.from_template(_SUMMARIZATION_PROMPT_TEMPLATE)


SUMMARIZATION_CHAIN = (
    RunnableLambda(lambda url: WebBaseLoader(url).load())
    | StuffDocumentsChain(
        llm_chain=LLMChain(llm=OPENAI_SUMMARIZATION_MODEL, prompt=_SUMMARIZATION_PROMPT),
        document_variable_name='text',
    )
    | RunnableLambda(lambda output: re.sub(r'\s+', ' ', output['output_text']))
)
