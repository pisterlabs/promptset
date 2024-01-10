from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import OpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List


model = OpenAI()


# Define your desired data structure.
class NationaLanguage(BaseModel):
    country: str = Field(description="question about country's language")
    spoken: str = Field(description="language of country")


choose_country = input("Pick a country ")
# And a query intented to prompt a language model to populate the data structure.
language_query = f"What language is spoken in {choose_country} ?"

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=NationaLanguage)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(query=language_query)

output = model(_input.to_string())

result = parser.parse(output)
print(result)
