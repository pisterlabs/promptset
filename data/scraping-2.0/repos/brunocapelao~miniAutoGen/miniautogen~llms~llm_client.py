import openai
import logging
from litellm import completion

# Interface for the LLM client
class LLMClientInterface:
    def get_model_response(self, prompt, model_name="gpt-4", temperature=1):
        """
        Retrieves a model response based on the given prompt.

        Args:
            prompt (str): The input prompt for the model.
            model_name (str, optional): The name of the model to use. Defaults to "gpt-4".
            temperature (float, optional): The temperature parameter for model generation. Defaults to 1.

        Returns:
            str: The generated model response.
        """
        raise NotImplementedError

# Concrete implementation for the OpenAI client
class OpenAIClient(LLMClientInterface):
    def __init__(self, api_key):
        """
        Initializes the OpenAI client.

        Args:
            api_key (str): The API key for accessing the OpenAI service.
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)

    def get_model_response(self, prompt, model="gpt-3.5-turbo", temperature=1):
        """
        Retrieves a model response using the OpenAI API.

        Args:
            prompt (str): The input prompt for the model.
            model (str, optional): The name of the model to use. Defaults to "gpt-3.5-turbo".
            temperature (float, optional): The temperature parameter for model generation. Defaults to 1.

        Returns:
            str: The generated model response.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error calling the OpenAI API: {e}")
            return None
        

class LiteLLMClient(LLMClientInterface):
    def __init__(self, model):
        """
        Initializes the LiteLLM client.

        Args:
            model: The LiteLLM model to use.
        """
        self.logger = logging.getLogger(__name__)
        self.model = model

    def get_model_response(self, prompt, model_name=None):
        """
        Retrieves a model response using the LiteLLM API.

        Args:
            prompt (str): The input prompt for the model.
            model_name (str, optional): The name of the model to use. Defaults to None.

        Returns:
            str: The generated model response.
        """
        try:
            response = completion(self.model, messages=prompt)
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error calling the LiteLMM API: {e}")
            return None
