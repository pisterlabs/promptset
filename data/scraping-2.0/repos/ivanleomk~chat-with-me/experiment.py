import openai
from openai_function_call import OpenAISchema, MultiTask

openai.api_key = "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"


class User(OpenAISchema):
    name: str
    age: int


MultiUser = MultiTask(User)

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    temperature=0.1,
    stream=False,
    functions=[MultiUser.openai_schema],
    function_call={"name": MultiUser.openai_schema["name"]},
    messages=[
        {
            "role": "user",
            "content": f"Consider the data below: Jason is 10 and John is 30",
        },
    ],
    max_tokens=1000,
)

import pdb

pdb.set_trace()
MultiUser.from_response(completion)
