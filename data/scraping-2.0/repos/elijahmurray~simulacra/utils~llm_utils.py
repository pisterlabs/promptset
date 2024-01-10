import openai
import dotenv
import os

from utils.logging_utils import Logger


dotenv.load_dotenv()

TEMPERATURE = 0.5
MODEL = "gpt-3.5-turbo"


class OpenAIManager:
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def chatCompletion(self, prompt: str, context: str = None) -> str:
        Logger.log("Querying OpenAI...", type="method")
        Logger.log("Context: \n" + str(context), type="context")
        Logger.log("\nPrompt: " + prompt, type="prompt")
        messages = []

        if context is not None:
            messages.append({"role": "system", "content": context})

        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=messages,
        )

        generated_response = response["choices"][0]["message"]["content"].strip()

        Logger.log(generated_response, type="openai_response")

        return generated_response
