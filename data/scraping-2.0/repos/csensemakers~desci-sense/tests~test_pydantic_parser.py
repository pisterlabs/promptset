import os
import sys
from pathlib import Path
ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))


from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

from desci_sense.parsers.base_parser import BaseParser
from desci_sense.postprocessing.parser_utils import fix_json_string_with_backslashes
from desci_sense.configs import init_config


# Define a new Pydantic model with field descriptions and tailored for Twitter.
class TwitterUser(BaseModel):
    name: str = Field(description="Full name of the user.")
    handle: str = Field(description="Twitter handle of the user, without the '@'.")
    age: int = Field(description="Age of the user.")
    hobbies: List[str] = Field(description="List of hobbies of the user.")
    email: str = Field(description="Email address of the user.")
    bio: str = Field(description="Bio or short description about the user.")
    location: str = Field(description="Location or region where the user resides.")
    is_blue_badge: bool = Field(
        description="Boolean indicating if the user has a verified blue badge."
    )
    joined: str = Field(description="Date the user joined Twitter.")
    gender: str = Field(description="Gender of the user.")
    appearance: str = Field(description="Physical description of the user.")
    avatar_prompt: str = Field(
        description="Prompt for generating a photorealistic avatar image. The image should capture the essence of the user's appearance description, ideally in a setting that aligns with their interests or bio. Use professional equipment to ensure high quality and fine details."
    )
    banner_prompt: str = Field(
        description="Prompt for generating a banner image. This image should represent the user's hobbies, interests, or the essence of their bio. It should be high-resolution and captivating, suitable for a Twitter profile banner."
    )
    
def test_pydantic_basic():
    # Instantiate the parser with the new model.
    parser = PydanticOutputParser(pydantic_object=TwitterUser)
    
    # Update the prompt to match the new query and desired format.
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                "answer the users question as best as possible.\n{format_instructions}\n{question}"
            )
        ],
        input_variables=["question"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        },
    )
    
    
    
    # Generate the input using the updated prompt.
    user_query = (
        "Generate a detailed Twitter profile of a random realistic user with a diverse background, "
        "from any country in the world, original name, including prompts for images. Come up with "
        "real name, never use most popular placeholders like john smith and john doe."
    )
    _input = prompt.format_prompt(question=user_query)
    
    
    config = init_config()
    nano_parser = BaseParser(config=config)
    chat_model = nano_parser.model
    
    
    output = chat_model(_input.to_messages())
    fixed_content = fix_json_string_with_backslashes(output.content)
    parsed = parser.parse(fixed_content)
    assert type(parsed) == TwitterUser
    
if __name__ == "__main__":
    # Instantiate the parser with the new model.
    parser = PydanticOutputParser(pydantic_object=TwitterUser)
    
    # Update the prompt to match the new query and desired format.
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                "answer the users question as best as possible.\n{format_instructions}\n{question}"
            )
        ],
        input_variables=["question"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        },
    )
    
    
    
    # Generate the input using the updated prompt.
    user_query = (
        "Generate a detailed Twitter profile of a random realistic user with a diverse background, "
        "from any country in the world, original name, including prompts for images. Come up with "
        "real name, never use most popular placeholders like john smith and john doe."
    )
    _input = prompt.format_prompt(question=user_query)
    print(_input.to_messages())
    
    
    config = init_config()
    nano_parser = BaseParser(config=config)
    chat_model = nano_parser.model
    
    
    output = chat_model(_input.to_messages())
    parsed = parser.parse(output.content)
    # print(output.content)
    # print(parsed)
