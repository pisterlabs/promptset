"""LLM interface."""

import json
import logging
import os
from typing import Literal

from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.llms.vertexai import VertexAI

from . import prompts

logging.basicConfig(level=logging.INFO)

stdout_handler = StdOutCallbackHandler()


class SnookerScoresLLM:
    """LLM client for extracting snooker scores from messages"""

    llms = {
        "openai": OpenAI,
        "vertexai": VertexAI,
    }

    def __init__(
        self,
        llm: Literal["openai", "vertexai"] = "openai",
        model_name=None,
        prompt=None,
    ):
        if model_name is None:
            self.llm = self.llms[llm]()
        else:
            self.llm = self.llms[llm](model_name=model_name)
        self.verbose = bool(os.getenv("LANGCHAIN_VERBOSE", False))
        if not prompt:
            prompt = prompts.get_prompt()
        self.prompt = prompt

    def infer(self, passage: str, players_blob: str) -> dict:
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt, verbose=self.verbose, callbacks=[stdout_handler])
        llm_output_raw = llm_chain.run(players_blob=players_blob, passage=passage)
        try:
            llm_output = json.loads(llm_output_raw) or {}
            logging.info(f"{self.llm.__class__.__name__} output: {llm_output}")
            output = {"passage": passage, **llm_output}
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not output valid JSON: {llm_output_raw}") from e
        return output
