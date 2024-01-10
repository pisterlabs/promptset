import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LanguageModelAPI:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = openai.OpenAI(api_key=self.api_key)

    def generate_text(self, prompt, max_tokens=150, temperature=0.7, top_p=1, n=1, stop=None, model="text-davinci-003"):
        """
        Generates text using the OpenAI API based on the given prompt and parameters.

        :param prompt: The input text prompt to generate content from.
        :param max_tokens: The maximum number of tokens to generate in the completion.
        :param temperature: Controls randomness. Lowering results in less random completions.
        :param top_p: An alternative to sampling with temperature, called nucleus sampling, where the model
                      considers the results of the tokens with top_p probability mass.
        :param n: Number of completions to generate.
        :param stop: One or more stop sequence which will end the completion.
        :param model: The model to use for the text generation.
        :return: The generated text as a string.
        """
        try:
            response = self.openai_client.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop
            )
            return response.choices[0].text.strip() if response.choices else ""
        except openai.error.OpenAIError as e:
            print(f"An error occurred: {e}")
            return ""

    def fine_tune_model(self, training_file, model="text-davinci-003"):
        """
        Fine-tunes the specified model on a given dataset.

        :param training_file: The path to the file containing the training data.
        :param model: The base model to fine-tune.
        :return: The ID of the fine-tuned model.
        """
        try:
            response = self.openai_client.FineTune.create(
                training_file=training_file,
                model=model
            )
            return response["id"]
        except openai.error.OpenAIError as e:
            print(f"An error occurred during fine-tuning: {e}")
            return ""

# Example usage:
# api = LanguageModelAPI()
# generated_text = api.generate_text("Once upon a time")
# print(generated_text)
