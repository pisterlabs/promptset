import logging
from typing import Any, Dict, List, Optional

import requests
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field

logger = logging.getLogger(__name__)


class LlamaCppApi(LLM):

    base_url: str
    """Base URL to the LLaMa example server http[s]://host:port """

    n_predict: Optional[int] = None
    """The maximum number of tokens to generate."""

    temperature: Optional[float] = 0
    """Primary factor to control randomness of outputs. 0 = deterministic
    (only the most likely token is used). Higher value = more randomness."""

    top_p: Optional[float] = 0.1
    """If not set to 1, select tokens with probabilities adding up to less than this
    number. Higher value = higher range of possible random results."""

    repeat_penalty: Optional[float] = 1.18
    """Exponential penalty factor for repeating prior tokens. 1 means no penalty,
    higher value = less repetition, lower value = more repetition."""

    top_k: Optional[float] = 40
    """Similar to top_p, but select instead only the top_k most likely tokens.
    Higher value = higher range of possible random results."""

    seed: int = Field(-1, alias="seed")
    """Seed (-1 for random)"""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling textgen."""
        return {
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "top_k": self.top_k,
            "seed": self.seed,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"base_url": self.base_url}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "llama_cpp_api"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        url = f"{self.base_url}/completion"
        request = self._default_params.copy()
        request["prompt"] = prompt
        response = requests.post(url, json=request)
        result = ""
        if response.status_code == 200:
            result = response.json()["content"]
        else:
            print(f"ERROR: response: {response}")

        return result
