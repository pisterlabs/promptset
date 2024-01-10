import unittest

from django.core.handlers.wsgi import WSGIRequest

import os
import openai


from ChatGptWX.chatitf import call_prompt


class MyTestCase(unittest.TestCase):
    response = openai.Completion.create(model="text-davinci-003", prompt="华为",
                                        temperature=0.5, stream=False,
                                        max_tokens=3048)
    print(response['choices'][0])


if __name__ == '__main__':
    unittest.main()
