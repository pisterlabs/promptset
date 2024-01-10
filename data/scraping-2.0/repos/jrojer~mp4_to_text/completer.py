from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai
import env

MODEL = "gpt-4-1106-preview"

openai.api_key = env.OPENAI_API_KEY


class Completer:
    def __init__(self, system_prompt: str, json_mode=False) -> None:
        self.system_prompt = system_prompt
        self.client = openai.OpenAI()
        self.temperature = 0.3
        self.json_mode = json_mode

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
    def complete(self, text: str) -> str:
        params = {
            "model": MODEL,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
        }
        if self.json_mode is True:
            params["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**params)
        return response.model_dump()["choices"][0]["message"]["content"]
