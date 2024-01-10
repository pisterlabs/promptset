from openai import OpenAI
from instructor import patch
from pydantic import Field, field_validator, BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = patch(client)

# "Code, schema, and prompt" We can run openai_schema to see exactly what the API will see
# Notice how the docstrings, attributes, types, and field descriptions are now part of the schema.

class UserDetails(BaseModel):
    "Correctly extracted user information"
    name: str = Field(..., description="User's full name")
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v.upper() != v:
            raise ValueError("Name must be capitalized")
        return v

# print(UserDetails.openai_schema)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    functions=[UserDetails.openai_schema],
    function_call={"name": UserDetails.openai_schema["name"]},
    max_retries=2,
    messages=[
        {"role": "system", "content": "Extract user details from my requests"},
        {"role": "user", "content": "My name is John Doe and I'm 30 years old."},
    ],
)

user = UserDetails.from_response(completion) # Deserialize back into a UserDetails object
print(user.name) # John Doe
print(user.age) # 30