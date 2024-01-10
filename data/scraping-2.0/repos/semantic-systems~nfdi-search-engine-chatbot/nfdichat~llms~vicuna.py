# -*- coding: utf-8 -*-
from typing import Any, Mapping

import requests
from langchain.llms.base import LLM

from nfdichat.common.config import llm_config


class VicunaLLM(LLM):
    config = llm_config["vicuna"]

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, **kwargs) -> str:
        """
        :param prompt:
        :param stop:
        :return:
        """
        response = requests.post(
            url=self.config["URL"],
            headers={"content-type": "application/json"},
            json={
                "model": self.config["MODEL_VERSION"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config["TEMPERATURE"],
                "key": self.config["KEY"],
            },
        )
        response.raise_for_status()
        return response.json()[0]["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
