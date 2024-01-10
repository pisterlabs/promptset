import intructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.patch(OpenAI())

class UserDetails:
    name: str
    age: int

user = client.chat.completions.create(
    model="gpt-3.5-turbo"
)
