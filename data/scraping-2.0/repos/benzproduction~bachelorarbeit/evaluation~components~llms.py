from typing import Any, Optional, List, Union, Callable, Iterable
from base import CompletionFn, CompletionResult, EmbeddingsFn, RetrieverFn
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import backoff
import openai
import pandas as pd
import string
import re

INVALID_STR = "__invalid__"
MATCH_FNS = {
    "include": lambda x, y: float(x in y),
    "exact": lambda x, y: float(x == y),
    "endswith": lambda x, y: x.endswith(y),
    "starts_or_endswith": lambda x, y: x.startswith(y) or x.endswith(y),
}

@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    result = openai.Completion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result

def formatSourcesDF(sources) -> str:
    """
    Helper function for formatting a pandas dataframe into a string for the Prompt.
    """
    if isinstance(sources, pd.DataFrame):
        assert "text" in sources.columns, "If sources are provided as a pandas dataframe, it must have a column named 'text'"
        assert "filename" in sources.columns or "key" in sources.columns, "If sources are provided as a pandas dataframe, it must have a column named 'filename' or 'key'"
        for i, row in sources.iterrows():
            if "key" in sources.columns:
                if row['text'].startswith(f"{row['key']}: "):
                    continue
                sources.loc[i,'text'] = f"{row['key']}: {row['text']};"
            else:
                if row['text'].startswith(f"{row['filename']}: "):
                    continue
                sources.loc[i,'text'] = f"{row['filename']}: {row['text']};"
        return sources['text'].str.cat(sep="\n\n")
    return sources

def formatSourcesList(sources) -> str:
    """
    Helper function for formatting a list of dicts into a string for the Prompt.
    """
    if isinstance(sources, list):
        if all(isinstance(source, str) for source in sources):
            return "\n\n".join(sources)
        assert all(
            isinstance(source, dict) and "text" in source and ("key" in source or "filename" in source)
            for source in sources
        ), "If sources are provided as a list of dicts, they must have keys 'text' and 'key' or 'filename'"
        for i, source in enumerate(sources):
            if "key" in source:
                sources[i]["text"] = f"{source['key']}: {source['text']};"
            else:
                sources[i]["text"] = f"{source['filename']}: {source['text']};"
        return "\n\n".join([source["text"] for source in sources])
    return sources

def format_necessary(template: str, allow_missing: bool = False, **kwargs: dict[str, str]) -> str:
    """Format a template string with only necessary kwargs."""
    keys = [k[1] for k in string.Formatter().parse(template) if k[1]]
    if allow_missing:
        assert (
            len([k for k in keys if k in kwargs]) > 0
        ), f"Required: {keys}, got: {sorted(kwargs)}, no inputs are used.\nTemplate:\n{template}"
        cur_keys = {k: kwargs.get(k, "{" + k + "}") for k in keys}
    else:
        assert all(
            k in kwargs for k in keys
        ), f"Required: {keys}, got: {sorted(kwargs)}.\nTemplate:\n{template}"
        cur_keys = {k: kwargs[k] for k in keys}
    return template.format(**cur_keys)

@dataclass
class Prompt(ABC):
    """
    A `Prompt` encapsulates everything required to present the `raw_prompt` in different formats.
    """

    @abstractmethod
    def to_formatted_prompt(self):
        """
        Return the actual data to be passed as the `prompt` field to your model.
        """

class CompletionPrompt(Prompt):
    def __init__(self, template: str, query: str, sources: Union[str, pd.DataFrame, list]):
        assert "{query}" in template, "Prompt template must contain {query}"
        assert "{sources}" in template, "Prompt template must contain {sources}"
        self.template = template
        self.query = query
        # Format sources
        if isinstance(sources, pd.DataFrame):
            sources = formatSourcesDF(sources)
        if isinstance(sources, list):
            sources = formatSourcesList(sources)
        if not isinstance(sources, str):
            raise ValueError(f"Sources must be a str, list, or pandas dataframe. Got {type(sources)}")
        self.sources = sources

    def to_formatted_prompt(self):
        return format_necessary(self.template, query=self.query, sources=self.sources)

class OpenAIBaseCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any, sources: Optional[Any] = None):
        self.raw_data = raw_data
        self.prompt = prompt
        self.sources = sources

    def get_completions(self) -> list[str]:
        raise NotImplementedError
    
    def get_sources(self) -> Optional[Any]:
        raise NotImplementedError

class OpenAICompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data and "choices" in self.raw_data:
            for choice in self.raw_data["choices"]:
                if "text" in choice:
                    completions.append(choice["text"])
                elif "message" in choice:
                    completions.append(choice["message"]["content"])
        return completions
    
    def get_sources(self) -> Optional[Any]:
        if isinstance(self.sources, pd.DataFrame):
            return formatSourcesDF(self.sources)
        return self.sources


class OpenAICompletionFn(CompletionFn):
    def __init__(
        self,
        api_key: str,
        api_base: str,
        deployment_name: Optional[str] = "text-davinci-003",
        api_type: Optional[str] = "azure",
        api_version: Optional[str] = "2022-12-01",
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.deployment_name = deployment_name
        self.api_type = api_type
        self.api_version = api_version
        self.extra_options = extra_options

    def __call__(
        self,
        prompt_template: str,
        query: str,
        sources: Optional[Union[str, pd.DataFrame, list]] = None,
        embedder: Optional[EmbeddingsFn] = None,
        retriever: Optional[RetrieverFn] = None,
        k: Optional[int] = 5,
        **kwargs,
    ) -> OpenAICompletionResult:
        assert sources or (isinstance(embedder, EmbeddingsFn) and isinstance(retriever, RetrieverFn)), "Either sources or an embedder and retriever must be provided"
        if not sources:
            sources = retriever(query, embedder, k=k)
        prompt = CompletionPrompt(template=prompt_template, query=query, sources=sources)
        result = openai_completion_create_retrying(
            engine=self.deployment_name,
            prompt=prompt.to_formatted_prompt(),
            api_key=self.api_key,
            api_base=self.api_base,
            api_type=self.api_type,
            api_version=self.api_version,
            **self.extra_options,
        )
        result = OpenAICompletionResult(raw_data=result, prompt=prompt, sources=sources)
        return result
    
    def __repr__(self):
        return f"OpenAICompletionFn(deployment_name={self.deployment_name}, extra_options={self.extra_options})"
    
class OpenAICompletion2StepFn(CompletionFn):
    def __init__(
        self,
        api_key: str,
        api_base: str,
        deployment_name: Optional[str] = "text-davinci-003",
        api_type: Optional[str] = "azure",
        api_version: Optional[str] = "2022-12-01",
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.deployment_name = deployment_name
        self.api_type = api_type
        self.api_version = api_version
        self.extra_options = extra_options
    
    def _get_choice(self, text: str, eval_type: str, match_fn: Union[str, Callable], choice_strings: Iterable[str]
        ) -> str:
        """Clean the answer string to a choice string to one of choice_strings. Return '__invalid__.' if no match."""
        if isinstance(match_fn, str):
            match_fn = MATCH_FNS[match_fn]
        lines = text.strip().split("\n")
        if eval_type.startswith("cot_classify"):
            lines = lines[::-1]  # reverse lines
        for line in lines:
            line = line.strip()
            line = "".join(c for c in line if c not in string.punctuation)
            if not line:
                continue
            for choice in choice_strings:
                if match_fn(line, choice):
                    return choice
        logging.warn(f"Choices {choice_strings} not parsable for {eval_type}: {text}")
        return INVALID_STR

    def __call__(
        self,
        step1_prompt: str,
        step2_prompt: str,
        query: str,
        sources: Optional[Union[str, pd.DataFrame, list]] = None,
        embedder: Optional[EmbeddingsFn] = None,
        retriever: Optional[RetrieverFn] = None,
        k: Optional[int] = 5,
        **kwargs,
    ) -> OpenAICompletionResult:
        assert sources or (isinstance(embedder, EmbeddingsFn) and isinstance(retriever, RetrieverFn)), "Either sources or an embedder and retriever must be provided"
        if not sources:
            sources = retriever(query, embedder, k=k)
        prompt1 = CompletionPrompt(template=step1_prompt, query=query, sources=sources)
        prompt2 = CompletionPrompt(template=step2_prompt, query=query, sources=sources)
        # STEP 1 asks the model to check weather the sources are enough to answer the query

        # STEP 2 asks the model to answer the query, but only if step 1 returned true
        # result = openai_completion_create_retrying(
        #     engine=self.deployment_name,
        #     prompt=prompt.to_formatted_prompt(),
        #     api_key=self.api_key,
        #     api_base=self.api_base,
        #     api_type=self.api_type,
        #     api_version=self.api_version,
        #     **self.extra_options,
        # )
        # result = OpenAICompletionResult(raw_data=result, prompt=prompt, sources=sources)
        # return result
    
    def __repr__(self):
        return f"OpenAICompletionFn(deployment_name={self.deployment_name}, extra_options={self.extra_options})"