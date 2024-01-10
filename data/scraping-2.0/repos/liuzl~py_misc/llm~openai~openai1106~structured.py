import instructor
from openai import OpenAI
from pydantic import BaseModel

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Enables `response_model`
client = instructor.patch(OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
))

class UserDetail(BaseModel):
    name: str
    age: int

user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ]
)

assert isinstance(user, UserDetail)
assert user.name == "Jason"
assert user.age == 25
print(user)
