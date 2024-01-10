###exaples
from helper import Helper

import openai


class OpenAPITest:

    # Results in:
    # {
    #   "role": "assistant",
    #   "content": "Kenobi."
    # }
    @staticmethod
    def test_open_api():
        openai.api_key = Helper().get_openapi_key()

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Star wars fan here, i am greeting with a quote, no deviations"},
                {"role": "user",
                 "content": "Hello there general"}
            ]
        )

        print(completion.choices[0].message)

    @staticmethod
    def return_result_from_open_api(sys_msg, user_msg):
        openai.api_key = Helper().get_openapi_key()

        completion = openai.Completion.create(
            engine="gpt-3.5-turbo",
          #  max_tokens=5
            messages=[
                {"role": "system",
                 "content": sys_msg},
                {"role": "user",
                 "content": user_msg}
            ]
        )

        print(completion.choices[0].text)
