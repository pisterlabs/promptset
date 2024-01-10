from typing import List

from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator
import constants


model_name = "text-davinci-003"
temperature = 0.0
model = OpenAI(openai_api_key=constants.APIKEY ,model_name=model_name, temperature=temperature)



# """ Define the data structure we want to be parsed out from the LLM response

# notice that the class contains a setup (a string) and a punchline (a string.
# The descriptions are used to construct the prompt to the llm. This particular
# example also has a validator which checks if the setup contains a question mark.

# from: https://python.langchain.com/docs/modules/model_io/output_parsers/pydantic
# """

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field
    
# """Defining the query from the user
# """
joke_query = "Tell me a joke about parrots"

# """Defining the prompt to the llm

# from: https://python.langchain.com/docs/modules/model_io/output_parsers/pydantic
# """
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(query=joke_query)

print(_input.text)

output = model(_input.to_string())

parser.parse(output)

# Joke(setup='Why did the chicken cross the road?', punchline='To get to the other side!')




