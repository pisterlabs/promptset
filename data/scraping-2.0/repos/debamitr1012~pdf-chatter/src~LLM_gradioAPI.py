from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from gradio_client import Client
class LLM_gradioAPI(LLM):
    """
    This class serves as a wrapper for using gradio APIs with LLMs in the langchain library.

    It allows you to interact with language models through Gradio APIs. 
    The attributes you need to specify are 'client_api' and 'api_name'.
    Attributes:
        client_api (str): The URL of the Gradio API endpoint.
        api_name (str): The name of the API to call on the Gradio server.
        n (int): words to retrieve.

    Methods:
        _llm_type: Returns the type of LLM, which is 'custom' for this wrapper.
        _call: Makes an API call to the Gradio server using the provided prompt and returns the response.
        _identifying_params: Returns identifying parameters used to differentiate LLM instances.
    """
    n: int
    client_api = ""
    api_name = ""
    @property
    def _llm_type(self) -> str:
        return "custom"
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """
        Make an API call to the Gradio server using the specified prompt and return the response.
        Parameters:
            prompt (str): The prompt or input text for the API call.
            stop (List[str], optional): List of stop words. Not used in this context.
            run_manager (CallbackManagerForLLMRun, optional): Callback manager for LLM run. Not used in this context.
        Returns:
            str: The response from the Gradio API.
        """
        # Remove the restriction on the stop parameter
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        # Make the API call using the Gradio Client
        client = Client(self.client_api)
        result = client.predict(prompt, api_name=self.api_name)
        # Return the response from the API
        return result
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}