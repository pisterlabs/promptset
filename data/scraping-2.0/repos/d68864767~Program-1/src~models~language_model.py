import openai
from src.api.openai_connector import OpenAIConnector
from src.utils.logger import Logger

class LanguageModel:
    def __init__(self, model_name="text-davinci-003"):
        """
        Initialize the LanguageModel with a specific OpenAI GPT model.

        :param model_name: The name of the model to be used for text generation.
        """
        self.model_name = model_name
        self.connector = OpenAIConnector()
        self.logger = Logger().get_logger()

    def generate_text(self, prompt, max_tokens=150, temperature=0.7, top_p=1.0, n=1, stop=None):
        """
        Generate text using the specified language model.

        :param prompt: The input text prompt to generate text from.
        :param max_tokens: The maximum number of tokens to generate in the completion.
        :param temperature: What sampling temperature to use. Higher values mean the model will take more risks.
        :param top_p: An alternative to sampling with temperature, called nucleus sampling, where the model
                      considers the results of the tokens with top_p probability mass.
        :param n: How many completions to generate for each prompt.
        :param stop: The sequence where the API will stop generating further tokens.
        :return: The generated text as a string or list of strings if n > 1.
        """
        try:
            return self.connector.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                model=self.model_name
            )
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            raise

    def fine_tune_model(self, training_data, model_name=None):
        """
        Fine-tune the language model on the provided training data.

        :param training_data: The data to fine-tune the model on.
        :param model_name: The name of the fine-tuned model to create.
        :return: The name of the fine-tuned model.
        """
        # This is a placeholder for fine-tuning functionality.
        # Actual implementation would require setting up and running the fine-tuning process
        # using OpenAI's fine-tuning API, which is beyond the scope of this code snippet.
        self.logger.info("Fine-tuning model...")
        # Code for fine-tuning would go here
        fine_tuned_model_name = model_name if model_name else f"{self.model_name}-fine-tuned"
        # Return the name of the fine-tuned model
        return fine_tuned_model_name
