import os

import openai
from django.test import TestCase
import environ


# Create your tests here.
class GetEnvTest(TestCase):

    def test_get_environment_variable(self):
        env = environ.Env(
            # set casting, default value
            DEBUG=(bool, False)
        )

        # Set the project base directory
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Take environment variables from .env file
        environ.Env.read_env(os.path.join(BASE_DIR, 'girlfriend_gpt/../girlfriend_gpt/../.env'))

        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')  # 기본 사용방법
        print(OPENAI_API_KEY)
        self.assertIsNotNone(OPENAI_API_KEY)


class OpenAiMessageTest(TestCase):

    def test_message_response(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=0.9,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            messages=[
                {"role":"system","content":'you are a test bot. answer include test'},
                {"role": "user", "content": "This is a test."},
            ],
        )
        print(response)
        self.assertIsNotNone(response)
        self.assertIs(type(response['choices'][0]['message']['content']), str)
