from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms import LlamaCpp


class CancellableLlamaCpp(LlamaCpp):
    stopRequested = False

    def stopGen(self):
        self.stopRequested = True

    def _call(
        self,
        prompt: str,
        stop: 'Optional[List[str]]' = None,
        run_manager: 'Optional[CallbackManagerForLLMRun]' = None,
        **kwargs: Any,
    ) -> str:
        # Modified implementation of LlamaCpp._call
        self.stopRequested = False
        if self.streaming:
            # If streaming is enabled, we use the stream
            # method that yields as they are generated
            # and return the combined strings from the first choices's text:
            combined_text_output = ""
            for token in self.stream(prompt=prompt, stop=stop, run_manager=run_manager):
                if self.stopRequested:
                    return combined_text_output
                combined_text_output += token["choices"][0]["text"]
            return combined_text_output
        else:
            params = self._get_parameters(stop)
            params = {**params, **kwargs}
            result = self.client(prompt=prompt, **params)
            return result["choices"][0]["text"]