import secrets
import openai

from configs.config import *
from configs.secrets import *


class GPT3:

    def __init__(self, engine="curie"):
        self._engine = engine
        openai.api_key = OPENAI

        with open("./prompts/james") as f:
            self._james_prompt = f.read()

        with open("./prompts/philosopher") as f:
            self._philosopher_prompt = f.read()

        with open("./prompts/dj") as f:
            self._dj_prompt = f.read()


    def ask_james(self, prompt):
        #print(self._james_prompt.replace("{!!!}", prompt))

        request = openai.Completion.create(
            engine=self._engine,
            temperature=0.7,
            prompt=self._james_prompt.replace("{!!!}", prompt),
            max_tokens=80,
            stop=["\n", "James"]
        )
        #print(request)

        completion = request.to_dict()['choices'][0]['text']
        return completion

    def ask_philosopher(self, prompt):

        request = openai.Completion.create(
            engine="davinci",
            temperature=0.9,
            prompt=self._philosopher_prompt.replace("{!!!}", prompt),
            presence_penalty=1,
            frequency_penalty=1,
            max_tokens=PHILO_MAX_TOKENS,
            best_of=1,
            stop=["Philosopher AI:", "Human:", "human:" ,"\n\n\n"] # , "\"\n", "\n\""]
        )

        print("GPT3.ask_philosopher request: ", request)

        completion = request.to_dict()['choices'][0]['text']
        return completion

    def ask_dj(self, prompt):
        request = openai.Completion.create(
            engine="davinci",
            temperature=0.9,
            prompt=self._dj_prompt.replace("{!!!}", prompt),
            presence_penalty=1,
            frequency_penalty=1,
            max_tokens=20,
            best_of=1,
            stop=["\n", "\n\n\n"] # , "\"\n", "\n\""]
        )

        print("GPT3.ask_philosopher request: ", request)

        completion = request.to_dict()['choices'][0]['text']
        return completion

if __name__ == "__main__":
    gpt = GPT3()
    print(gpt.ask_dj("Song for hyping me up"))
