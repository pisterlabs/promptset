from config import OPENAI_MODEL

import openai

SUMMARIZE_RUS = "Подведи итог разговора в 5 словах или меньше"
SUMMARIZE_ENG = "Summarize the conversation in 5 words or fewer"


class Session:

    def __init__(self, context=None, virgin=True, lang='ru', title=None):
        if context is None:
            context = []
        self.context = context
        self.lang = lang
        self.virgin = virgin
        self.title = title

    def new_message(self, text):
        response = Session.ask_ai(text, self.context)
        if response == -1:
            return -1
        self.context.append({"role": "user", "content": text})
        self.context.append({"role": "assistant", "content": response})
        if self.virgin:
            self.create_title()
        return response

    def create_title(self):
        summarize = SUMMARIZE_RUS
        if self.lang == 'en':
            summarize = SUMMARIZE_ENG
        title = Session.ask_ai(summarize, self.context)
        self.title = title
        self.virgin = False

    @staticmethod
    def ask_ai(prompt, context=None):
        if context is None:
            context = []
        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=context + [{"role": "user", "content": prompt}],
            )
        except openai.error.RateLimitError:
            return -1
        return response['choices'][0]['message']['content']
