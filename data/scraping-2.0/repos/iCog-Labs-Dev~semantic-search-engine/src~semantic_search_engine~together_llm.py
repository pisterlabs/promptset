from typing import Any, Dict #, List, Mapping, Optional
import together
from pydantic import Extra, root_validator #, Field
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from semantic_search_engine.constants import TOGETHER_API_KEY, TOGETHER_MODEL_NAME


class TogetherLLM(LLM):
    """A custom langchain LLM wrapper for togetherAI
    """

    model: str = TOGETHER_MODEL_NAME
    together_api_key: str = TOGETHER_API_KEY

    temperature: float = 2
    max_tokens: int = 1024

    # temperature: float = 0.7
    # max_tokens: int = 512

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        try:
            output = together.Complete.create(prompt,
                                            model=self.model,
                                            max_tokens=self.max_tokens,
                                            temperature=self.temperature,
                                            )
            text = output['output']['choices'][0]['text']
        except together.error.InstanceError:
            return f'The model "{ self.model }" is not running on together.ai'
        except:
            return 'An error occurred!'
        
        return text

    def start(self):
        together.Models.start(self.model)
    def stop(self):
        together.Models.stop(self.model)