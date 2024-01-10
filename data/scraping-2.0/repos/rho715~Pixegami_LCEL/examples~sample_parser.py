from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from packages.models import CHAT_LLM, CHAT_LLM_4
from langchain.output_parsers import PydanticOutputParser

# Define a Pydantic model
class Movie(BaseModel):
    title: str = Field(description="Title of the movie")
    genre: list[str] = Field(description="Genre of the movie")
    year: int = Field(description="Year of the movie")

# Set up the models
llm3 = CHAT_LLM
llm4 = CHAT_LLM_4

# Set up the output parser
parser = PydanticOutputParser(pydantic_object=Movie)

# Set up the prompt template
prompt_template_text = """
Response with a movie recommendation based on the query:\n
{format_instructions}\n
{query}
"""

# Set up the format instructions & prompt template
format_instructions = parser.get_format_instructions()
prompt_template = PromptTemplate(
    template=prompt_template_text,
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

# using lcel - GPT3
chain3 = prompt_template | llm3 | parser
response = chain3.invoke({"query": "A 90s movie with Leonardo DiCaprio."})
print(f"Response for GPT3: {response}")

# using lc - GPT4
chain4 = prompt_template | llm4 | parser
response = chain4.invoke({"query": "A 90s movie with Leonardo DiCaprio."})
print(f"Response for GPT4: {response}")


