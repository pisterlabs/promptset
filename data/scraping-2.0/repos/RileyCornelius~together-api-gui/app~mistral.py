import os
from typing import Iterator
import openai
import dotenv
import termcolor


class Mistral:
    def __init__(self, together_api_key: str = None, system_prompt: str = "", enable_print: bool = True):
        self.system_prompt = system_prompt
        self.enable_print = enable_print
        self.max_tokens = 1024
        self.temperature = 0.7
        self.top_p = 0.7
        self.model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

        if together_api_key is None:
            dotenv.load_dotenv()
            together_api_key = os.getenv("TOGETHER_API_KEY")
        self._client = openai.OpenAI(api_key=together_api_key, base_url="https://api.together.xyz/v1")
        self._history = []

    def chat(self, prompt: str) -> str:
        messages = self._build_prompt(prompt)
        output = self._client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        self._total_tokens = output.usage.total_tokens
        response = output.choices[0].message.content
        if self.enable_print:
            print(termcolor.colored("User: ", "cyan") + prompt)
            print(termcolor.colored("Assistant: ", "yellow") + response)
        self._append_history(prompt, response)
        return response

    def chat_stream(self, prompt: str) -> Iterator[str]:
        messages = self._build_prompt(prompt)
        stream = self._client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True,
        )
        if self.enable_print:
            print()
            print(termcolor.colored("User: ", "cyan") + prompt)
            print(termcolor.colored("Assistant:", "yellow"), end="")

        output = ""
        for chunk in stream:
            text = chunk.choices[0].delta.content
            output += text
            if self.enable_print:
                print(text or "", end="", flush=True)
            yield text

        self._append_history(prompt, output)

    def clear_history(self):
        self._history = []

    def _build_prompt(self, user_input: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        for pair in self._history:
            messages.append({"role": "user", "content": pair[0]})
            messages.append({"role": "assistant", "content": pair[1]})
        messages.append({"role": "user", "content": user_input})
        return messages

    def _append_history(self, user_input: str, model_output: str):
        self._history.append([user_input, model_output])


if __name__ == "__main__":
    mistral = Mistral(system_prompt="Always end your response with the word TERMINATE.")
    mistral.chat("Hello, how are you?")
    strea = mistral.chat_stream("Tell me more")
    for chunk in strea:
        pass
