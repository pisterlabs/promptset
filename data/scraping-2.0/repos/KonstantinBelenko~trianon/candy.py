# This file contains a set of wrappers for OpenAI's API.
# The wrappers are designed to be used in a chatbot context
#
# Example 1 - Completion
# response = Candy.completion() \
#     .set_prompt("Trianon is a very cool company that provides ai-infused customer support assistance to enterprise clients. \n\nWhat is Trianon?") \
#     .set_max_tokens(256) \
#     .set_engine("text-davinci-003") \
#     .set_stop(["stop"]) \
#     .run() \
#     .strip()
# print(response)

# Example 2 - Chat
# response = Candy.chat() \
#     .set_prompt("user", "What is Trianon?") \
#     .set_prompt("system", "Trianon is a very cool company that provides ai-infused customer support assistance to enterprise clients") \
#     .set_prompt("user", input("> ")) \
#     .run()
# print(f"Role: {response['role']}: {response['content']}")

from nougat import SimilarConcept
import openai
import os

class Runner:

    def __init__(self) -> None:
        self.prompt = ""
        self.engine = 'text-davinci-003'
        self.openai = openai
        self.openai.api_key = os.getenv("OPENAI_API_KEY")
    
class CompletionRunner(Runner):

    def __init__(self) -> None:
        super().__init__()
        self.output = ""
        self.tokens = 128
        self.temperature = 0.7
        self.separator = ""
        self.append_prompt = False

    def set_prompt(self, prompt: str):
        self.prompt = prompt
        return self

    def set_max_tokens(self, tokens: int):
        self.tokens = tokens
        return self
    
    def append_input(self, separator: str = ""):
        self.append_prompt = True
        self.separator = separator
        return self
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature
        return self

    def set_stop(self, stop: list):
        self.stop = stop
        return self
    
    def set_engine(self, engine: str):
        self.engine = engine
        return self

    def run(self):
        result = openai.Completion.create(
            engine=self.engine,
            prompt=self.prompt,
            temperature=self.temperature,
            max_tokens=self.tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=self.stop
        )

        text = result.choices[0].text
        if self.append_prompt:
            text = self.prompt + self.separator + text

        self.output = text

        return text

class ChatRunner(Runner):

    def __init__(self) -> None:
        super().__init__()
        self.message_log = []
        self.engine = 'gpt-3.5-turbo'

    def set_prompt(self, role: str, prompt: str):
        self.message_log.append({'role': role, 'content': prompt})
        self.prompt = prompt
        return self

    def run(self) -> str:
        response = openai.ChatCompletion.create(
            model=self.engine,
            messages=self.message_log
        )

        self.set_prompt(response.choices[-1].message.content, response.choices[-1].message.role)

        return {
            'role': response.choices[-1].message.role,
            'content': response.choices[-1].message.content
        }

class Candy:

    def __init__(self) -> None:
        pass        

    @staticmethod
    def completion():
        return CompletionRunner()
    
    @staticmethod
    def chat():
        return ChatRunner()
        
    @staticmethod
    def prepare_similar_query(query: str, concepts: list[SimilarConcept]):

        final_prompt = ""
        for similar in concepts:
            final_prompt += f"{similar.concept}\n"
            final_prompt += f"{similar.description}\n\n"

        final_prompt += f"{query}\n"
        return final_prompt