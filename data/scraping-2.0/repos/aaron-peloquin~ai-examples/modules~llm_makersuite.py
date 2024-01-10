from typing import Any, List, Mapping, Optional
from dotenv import dotenv_values
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import re

import google.generativeai as palm

env_config = dotenv_values(".env")
palm.configure(api_key=env_config["MAKER_KEY"])

class MakerSuite(LLM):
    max_output_tokens: int = 800
    
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        google_call = palm.generate_text(
            model = "models/text-bison-001",
            prompt = prompt,
            max_output_tokens = self.max_output_tokens,
            stop_sequences=stop
        )
        output_text = google_call.result
        if output_text is None:
            raise ValueError("Error generating text from API")
        return output_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_output_tokens": self.max_output_tokens,
        }
