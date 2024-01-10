import os
import openai
from dotenv import load_dotenv

load_dotenv()


class Assistant:
    def __init__(self):
        self.messages = []
        self.prompt = self.fetch_prompt()
        self.meta_setup()
        print("GPT CONNECTED -- ")

    def fetch_prompt(self):
        dat = None
        with open("api/prompt.in", "r") as inp:
            dat = inp.read()
        return dat

    def write_message(self, role: str, content: str) -> str:
        to_write = {"role": role, "content": content}
        self.messages.append(to_write)
        api_response = self.get_api_response()
        api_response_message = self.get_response_message(api_response)
        api_response_content = self.get_response_content(api_response)
        self.update_messages(api_response_message)

        return api_response_content

    def update_messages(self, message) -> None:
        self.messages.append(message)

    def meta_setup(self, use_gameshow_prompt=True) -> None:
        openai.api_key = os.getenv("API_KEY")
        self.write_message(
            role="system",
            content=self.prompt
            if use_gameshow_prompt
            else "You are a helpful assistant.",
        )

    def get_api_response(self):
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=self.messages
        )

    def get_response_message(self, resp):
        try:
            return resp["choices"][0]["message"]
        except:
            raise ValueError("Failed to get response message from response object")

    def get_response_content(self, resp):
        try:
            return resp["choices"][0]["message"]["content"]
        except:
            raise ValueError("Failed to get response content from response object")


if __name__ == "__main__":
    gpt = Assistant()
    print(gpt.prompt)
