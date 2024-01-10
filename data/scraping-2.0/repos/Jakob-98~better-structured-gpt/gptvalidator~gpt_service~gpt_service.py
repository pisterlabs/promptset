from typing import List, Dict
import logging
import openai

class GptService():
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

    def _validate_input(self, messages: List[Dict[str, str]], temperature: float) -> None:
        self.logger.info("Validating input parameters.")
        
        if not isinstance(messages, list):
            self.logger.error("Input messages must be a list.")
            raise TypeError("Input messages must be a list.")
        
        for message in messages:
            if not isinstance(message, dict):
                self.logger.error("Each message in the list must be a dictionary.")
                raise TypeError("Each message in the list must be a dictionary.")
                
            if 'role' not in message or 'content' not in message:
                self.logger.error("Each message must contain 'role' and 'content'.")
                raise ValueError("Each message must contain 'role' and 'content'.")
        
        if not isinstance(temperature, (int, float)):
            self.logger.error("Temperature must be a number.")
            raise TypeError("Temperature must be a number.")
        
        if temperature < 0 or temperature > 1:
            self.logger.error("Temperature must be in the range [0, 1].")
            raise ValueError("Temperature must be in the range [0, 1].")


    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        self._validate_input(messages, self.temperature)

        try:
            self.logger.info("Sending request to OpenAI API.")
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            content = response["choices"][0]["message"]["content"]
            # content = response
            self.logger.info(f"Response received successfully: {content}")
            return content
        except Exception as e:
            self.logger.error("An error occurred while retrieving response from OpenAI API: %s", str(e))
            raise
