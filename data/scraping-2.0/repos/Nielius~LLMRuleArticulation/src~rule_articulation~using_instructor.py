import instructor
from openai import OpenAI
from pydantic import BaseModel

from rule_articulation.secrets import get_openai_key

get_openai_key()

# Enables `response_model`
client = instructor.patch(OpenAI(**get_openai_key()))

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