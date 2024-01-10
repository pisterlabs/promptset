import asyncio
import time
import typing
import openai


class GPTApiRequest:
    """asks the chosen question to gpt, with the code, and returns the answer"""
    def __init__(self, TOKEN: str) -> None:
        self.__TOKEN: str = TOKEN
        self.__url: str = "https://api.openai.com/v1/engines/davinci-codex/completions"
        self.__response = []

    def __request(self, question: str) -> str:
        try:
            openai.api_key = self.__TOKEN
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=question,
                max_tokens=1000
            )
            print(response)
            time.sleep(21)  # necessario esperar 28s , do contrário, timeout.
            print(f"[+] OK")
            return response["choices"][0]["text"]
        except Exception as e:
            return f"Fail to connect: {e}"

    def ask(self, question: str) -> str:
        return self.__request(question)


if __name__ == '__main__':
    gpt = GPTApiRequest("token")
    print(gpt.ask("what your name?"))
