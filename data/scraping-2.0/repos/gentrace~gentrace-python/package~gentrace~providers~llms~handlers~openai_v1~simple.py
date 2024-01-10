from typing import cast

from openai import OpenAI

from gentrace.providers.init import GENTRACE_CONFIG_STATE
from gentrace.providers.llms.openai_v1 import GentraceAsyncOpenAI, GentraceSyncOpenAI


class SimpleGentraceSyncOpenAI(GentraceSyncOpenAI):
    def __init__(self, *args, **kwargs):
        if not GENTRACE_CONFIG_STATE["GENTRACE_API_KEY"]:
            raise ValueError(
                "No Gentrace API key available. Please use init() to set the API key."
            )

        gentrace_config = {
            "api_key": GENTRACE_CONFIG_STATE["GENTRACE_API_KEY"],
            "host": GENTRACE_CONFIG_STATE["GENTRACE_BASE_PATH"],
        }

        super().__init__(*args, **kwargs, gentrace_config=gentrace_config)


SimpleGentraceSyncOpenAITyped = cast(OpenAI, SimpleGentraceSyncOpenAI)


class SimpleGentraceAsyncOpenAI(GentraceAsyncOpenAI):

    def __init__(self, *args, **kwargs):
        if not GENTRACE_CONFIG_STATE["GENTRACE_API_KEY"]:
            raise ValueError(
                "No Gentrace API key available. Please use init() to set the API key."
            )

        gentrace_config = {
            "api_key": GENTRACE_CONFIG_STATE["GENTRACE_API_KEY"],
            "host": GENTRACE_CONFIG_STATE["GENTRACE_BASE_PATH"],
        }

        super().__init__(*args, **kwargs, gentrace_config=gentrace_config)


SimpleGentraceAsyncOpenAITyped = cast(OpenAI, SimpleGentraceAsyncOpenAI)

__all__ = [
    "SimpleGentraceSyncOpenAITyped",
    "SimpleGentraceAsyncOpenAITyped"
]
