import os
from functools import partial
from string import Formatter
from typing import Any

from dotenv import load_dotenv
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from langchain.chat_models import ChatOpenAI  # noqa: F401
from langchain.prompts import PromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables import ConfigurableField

from src.services.io import count_tokens

load_dotenv()

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "langchain-tracing2"

# UNCOMMENT FOR FAKE LLM
# from langchain.chat_models import FakeListChatModel  # noqa: E402

# llm = FakeListChatModel(responses=["foo", "bar", "baz"])

# # LLM DEBUGGER - PROVIDES VERBOSE READOUT OF LLM OBJECT CONFIG AND PROMPT AT INVOCATION
# from langchain.callbacks.base import BaseCallbackHandler
# class LLMTempReporter(BaseCallbackHandler):
#     def on_llm_start(
#         self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
#     ) -> Any:
#         print(f"LLM START WITH PROPERTIES {serialized} AND PROMPTS {prompts} AND KWARGS {kwargs}")

# temp_reporter = LLMTempReporter()

# # ADD AS LLM ARGUMENT FOR DEBUG
# llm = ChatOpenAI(callbacks=[temp_reporter]).configurable_fields(

# # UNCOMMENT FOR LIVE OPENAI LLM
llm = ChatOpenAI().configurable_fields(
    openai_api_key=ConfigurableField(
        id="api_key",
        name="OpenAI API Key",
        description="An API key for OpenAI. Get one from https://openai.com",
    ),
    openai_organization=ConfigurableField(
        id="organization",
        name="OpenAI Organization",
        description="For users who belong to multiple organizations, you can pass a header to specify which organization is used for an API request. Usage from these API requests will count as usage for the specified organization."
    ),
    model_name=ConfigurableField(
        id="model",
        name="OpenAI Model",
        description="The model to use for LLM calls. Defaults to gpt-3.5-turbo",
    ),
    temperature=ConfigurableField(
        id="temperature",
        name="LLM Temperature",
        description="Controls randomness of the output. Values closer to 0 make output more random, values closer to 1 make output more deterministic. If not specified, default is 0.7",
    )
)


def safe_format(template: str, **kwargs) -> str:
    """
    Safely formats the given template string using the provided keyword arguments.
    If a key in the template is not found in kwargs, it leaves the placeholder as is.

    Args:
        template (str): The template string to format.
        **kwargs: Keyword arguments for formatting.

    Returns:
        str: The formatted string.
    """

    class SafeDict(dict):
        @staticmethod
        def __missing__(key):
            return "{" + key + "}"

    formatter = Formatter()
    safe_kwargs = SafeDict(**kwargs)

    return formatter.vformat(template, (), safe_kwargs)


prompts = {
    "core_prompt_template": str("Summarize {page_content}"),
    "collapser_prompt_template": str("Collapse {context}"),
    "combiner_prompt_template": str("Combine these summaries:\n\n{context}"),
}  # always redefined at runtime.


def populate_template(dict_of_substitutions: dict, template: str) -> str:
    """Populate the template with the given substitutions."""
    return safe_format(template, **dict_of_substitutions)


# format_document is included in chain rather than partialed as a global variable because inital_prompt is redefined at runtime.
map_chain = (
    {
        "doc_in_template": lambda x: format_document(
            x, prompt=PromptTemplate.from_template(prompts["core_prompt_template"])
        )
    }
    | PromptTemplate.from_template("{doc_in_template}")
    | llm
    | StrOutputParser()
).with_config(run_name="Summarize (return doc)")

sum_and_recombine = (
    RunnableParallel({"doc": RunnablePassthrough(), "content": map_chain})
    | (lambda x: Document(page_content=x["content"], metadata=x["doc"].metadata))
).with_config(run_name="Summaries to Document")


# The chain we'll repeatedly apply to collapse subsets of the documents
# into a consolidate document until the total token size of our
# documents is below some max size.
def combine_docs(docs) -> str:
    return "\n\n".join(
        f"{doc.page_content}"
        for doc in docs
        # f"metadata: {doc.metadata}\ncontent: {doc.page_content}" for doc in docs
    )


collapse_chain: Runnable[Any, str] = (
    {
        "docs_in_template": {"context": combine_docs} | RunnableLambda(
            lambda x: populate_template(
                x, template=prompts["collapser_prompt_template"]
            )
        )
    }
    | PromptTemplate.from_template("{docs_in_template}")
    | llm
    | StrOutputParser()
)


# TODO: use the custom get_num_tokens fn w llm argument passed in? Or refactor custom fn as wrapper?
# LC method ties count to LLM which is desirable IF it actually does...
# ... looked hardcoded in my review of the code. Check this.
def get_num_tokens(docs: list[Document]) -> int:
    return count_tokens(combine_docs(docs))


async def _collapse(
    docs,
    config,
    collapse_token_max=6000,
    iteration_limit=2,
):
    collapse_ct = 1

    collapse_token_max = (
        config["attribute_configs"]["collapse_token_max"]
        if "attribute_configs" in config
        and "collapse_token_max" in config["attribute_configs"]
        else collapse_token_max
    )
    iteration_limit = (
        config["attribute_configs"]["iteration_limit"]
        if "attribute_configs" in config
        and "iteration_limit" in config["attribute_configs"]
        else iteration_limit
    )

    while get_num_tokens(docs) > collapse_token_max and collapse_ct < iteration_limit:

        # configure collapse_chain to include run number
        config["run_name"] = f"Collapse {collapse_ct}"
        collapse_chain_w_config = partial(collapse_chain.invoke, config=config)

        # create a list of lists of docs, each with content (excl. metadata) no longer than token_max (pops docs from queue until max, doesn't mix and match)
        split_docs = split_list_of_docs(docs, get_num_tokens, collapse_token_max)

        # execute collapse_chain on each list of docs
        docs = [collapse_docs(_docs, collapse_chain_w_config) for _docs in split_docs]

        collapse_ct += 1

    return docs


collapse = RunnableLambda(_collapse)

# The chain we'll use to combine our individual document summaries
# (or summaries over subset of documents if we had to collapse the map results)
# into a final summary.

reduce_chain: Runnable[Any, str] = (
    {
        "docs_in_template": {"context": combine_docs} | RunnableLambda(
            lambda x: populate_template(x, template=prompts["combiner_prompt_template"])
        )
    }
    | PromptTemplate.from_template("{docs_in_template}")
    | llm
    | StrOutputParser()
).with_config(run_name="Reduce")


# The final full chain
map_reduce_chain = (
    sum_and_recombine.with_config(run_name="Map").map() | collapse | reduce_chain
).with_config(run_name="Map reduce")


async def map_reduce(
    docs: list[Document],
    core_prompt: str,
    reduce_prompt: str | None = None,
    combine_summaries_prompt: str | None = None,
    *,
    api_key: str | None = None,
    organization: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_concurrency: int = 3,
    **kwargs,
) -> str:
    """Summarize a list of documents. The documents are first summarized individually
    using the core_prompt. Then those summaries are combined and summarized using
    the reduce_prompt. The reduced summaries are combined using the combine_summaries_prompt.

    Args:
        docs: A list of documents to summarize
        core_prompt: The prompt to use for the individual Document summarization.
            Must include {page_content} where the content of the document will be inserted.
        reduce_prompt: The prompt to use for the summary reduction.
            If None, the default is:\n\n
            "The summary that follows was generated based on the following prompt:\n\n
            {core_prompt}\n\n
            I want you to follow the same instructions, but
            shorten the summary still further. Here's the summary:\n\n"
        combine_summaries_prompt: The prompt to use for the summary combination.
            If None, the default is:\n\n
            "The summaries that follow were generated based on the following prompt:\n\n
            {core_prompt}\n\n
            I want you to follow the same instructions while combining
            the summaries.Here are the summaries to combine:\n\n"
        max_concurrency: The maximum number of concurrent requests to allow.

    Returns:
        The summary as a string.

    """

    # Build configuration map
    configurables = {}

    if api_key is not None:
        configurables["api_key"] = api_key
    if organization is not None:
        configurables["organization"] = organization
    if model is not None:
        configurables["model"] = model
    if temperature is not None:
        configurables["temperature"] = temperature

    attribute_configs = {}

    attribute_configs.update(kwargs)

    callbacks = [ConsoleCallbackHandler()]

    max_concurrency = max_concurrency

    combined_configs = {
        "callbacks": callbacks,
        "configurable": configurables,
        "attribute_configs": attribute_configs,
        "max_concurrency": max_concurrency,
    }

    # set prompts
    global prompts
    prompts["core_prompt_template"] = str(core_prompt)

    if reduce_prompt is None:
        prompts["collapser_prompt_template"] = str(
            "The summary that follows was generated based on the following prompt:\n\n"
            + core_prompt
            + "\n\nI want you to follow the same instructions, but shorten the summary"
            " while being very careful not to cut information the user wants."
            " If you can't shorten the summary "
            " without violating the user's instructions, then just repeat the same"
            " summary back to the user. Here's the summary:\n\n{context}"
        )
    else:
        _formatted_reduce_prompt = reduce_prompt.format(
            core_prompt=prompts["core_prompt_template"]
        )
        prompts["collapser_prompt_template"] = str(
            _formatted_reduce_prompt + "{context}"
        )

    if combine_summaries_prompt is None:
        prompts["combiner_prompt_template"] = str(
            "The summaries that follow were generated based on the following"
            " prompt:\n\n"
            + core_prompt
            + "\n\nI want you to follow the same instructions while combining the"
            " summaries. You should remove metadata and duplicate information, and"
            " format the output, but do not cut out information. Here are the summaries"
            " to combine:\n\n{context}"
        )
    else:
        _formatted_combine_prompt = combine_summaries_prompt.format(
            core_prompt=prompts["core_prompt_template"]
        )
        prompts["combiner_prompt_template"] = str(
            _formatted_combine_prompt + "{context}"
        )

    # run chain
    result = await map_reduce_chain.with_config(combined_configs).ainvoke(docs)
    # result = await map_reduce_chain.ainvoke(docs, config=combined_configs)

    print(result)

    return result
