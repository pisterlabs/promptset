import openai

from llm.util import *
from llm.openai_utils import *
import time


class SentiChat(openai.ChatCompletion):

    def __init__(self, debug=False):
        super().__init__(engine="gpt-3.5-turbo")
        self.debug = debug
        self.engine = "gpt-3.5-turbo"
        self.intro = [{"role": "system",
                       "content": "You are a sentiment analyzer.  You will respond to a user's prompt with a two "
                                  "four part response seperated by '|'.  The first part will be a sentiment score "
                                  "between -1 and 1.  -1 means the user's prompt is very negative, while 1 means the "
                                  "user's prompt is very positive, a score of 0 is neutral. The second part is your "
                                  "confidence in the rating with 0 being no confidence and 1 being extremely confident."
                                  "The forth part will be a one to two sentence explanation of your reasoning for "
                                  "rating a text as positive, negative or neutral. The third part will be your grade of"
                                  "the explanation with 0 being the worst and 1 being the best. The overall response "
                                  "response format is: Sentiment Score [-1 - 1] | Confidence Rating [0 -1] | "
                                  "Explanation Grade [0 - 1] | Explanation [Free Text]"}]
        self.messages = self.intro
        self.prompt = "Analyze the sentiment of the following:"
        self.response = None

    def create(self, text):
        response = super().create(
            model=self.engine,
            messages=text,
        )
        if self.debug:
            yellow(response)
        return response["choices"][0]["message"]

    def say_last(self):
        cyan(self.messages[-1]["content"])

    def get_input(self, text):
        pass

    def classify_sentiment(self, text, verbose=False):
        if self.debug:
            yellow(text)
        self.messages.append({"role": "user", "content": self.prompt + text})
        if self.debug:
            cyan(self.messages)
        max_retries = 5
        retries = 0
        while retries < max_retries:
            try:
                self.response = self.create(self.messages)
                retries = 5
            except openai.OpenAIError as e:
                red(f"An error occurred: {e}")
                retries += 1
                red(f"Retrying ({retries}/{max_retries})...")
                time.sleep(1)  # Wait for a short period before retrying
        if verbose:
            self.say_last()
        return self.response['content']


class SentiSummary(SentiChat):

    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.engine = "gpt-3.5-turbo"
        self.intro = [{"role": "system",
                       "content": "You a component of a sentiment analyzer,  your only job is to summarize a list of "
                                  "explanations into a single explanation. In essence you are averaging five different "
                                  "explanations to derive a summary explanation"}]
        self.messages = self.intro
        self.prompt = "Provide a 1 - 2 sentence summary of the following explanations:"
        self.response = None

    def summarize_explanations(self, explanations, verbose=False):
        if self.debug:
            red(explanations)
        summary = self.classify_sentiment(explanations, verbose=verbose)
        return summary


def create_senti_chat_bot():
    openai.api_key = get_openai_key()
    cb = SentiChat()
    cb.say_last()
    run = True
    while run:
        cb.chat()
        if cb.messages[-1]["content"] == "exit":
            run = False

# create_senti_chat_bot()
