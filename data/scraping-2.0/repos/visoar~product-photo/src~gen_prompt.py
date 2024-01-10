# @title gen_prompt_with_gpt
import threading

import openai

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return


def gen_prompt_with_gpt(api_key: str, prompt: str):
    openai.api_key = api_key
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": """
You are a professional AI image caption generator,
generates a 30 words image scene in english based on user input.

1. Scene format: [Product] [in | on | close by | on top of | below | ...] [something], ...
2. Increase/Decrease thing weight of by using the format {importance thing}+/{background}-

User input: chair
Scene in english: chair+ in a sunny modern living room-, filled with leafy plants+

User input:`""" + prompt + """`
Scene in english:
"""}
        ]
    )
    prompt = completion.choices[0].message.content
    print('  Image prompt: ' + prompt)
    return prompt


def gen_prompt(api_key: str, prompt: str):
    return ThreadWithReturnValue(target=gen_prompt_with_gpt, args=(api_key, prompt,))
