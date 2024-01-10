import bardapi
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class BardLLM(LLM):
    token: str

    @property
    def _llm_type(self) -> str:
        """
        Get the type of this LLM (Language Model) instance.
        
        Returns:
            str: The type of this LLM, which is 'custom_bard'.
        """
        return "custom_bard"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response using the BARD LLM (Language Model) based on the given prompt.

        Args:
            prompt (str): The input prompt for generating a response.
            stop (Optional[List[str]]): A list of strings indicating stop conditions (not used in this method).
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager for LLM run (not used in this method).
            **kwargs (Any): Additional keyword arguments (not used in this method).

        Returns:
            str: The generated response from the BARD LLM.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response = bardapi.core.Bard(self.token).get_answer(prompt)
        res = response["content"]
        print(f"Response from BARD LLM: {res}")
        return res

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Get the identifying parameters of this LLM instance.

        Returns:
            Mapping[str, Any]: A mapping of identifying parameters, which includes the 'token' attribute.
        """
        return {"token": self.token}
