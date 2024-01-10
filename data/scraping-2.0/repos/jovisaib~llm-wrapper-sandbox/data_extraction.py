from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator





class Movie(BaseModel):
    title: str = Field(description="the title of the movie")
    year: str = Field(description="year of the movie")
    director: str = Field(description="the name of the director of the movie")

model = OpenAI(temperature=0.0)

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = PromptTemplate(
    template="Recommend the most similar movie to this one (ignore sequels or prequels): {title}\n{format_instructions}\n",
    input_variables=["title"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def get_similar_movie(title: str) -> Movie:
    _input = prompt.format_prompt(title=title)
    output = model(_input.to_string())
    return parser.parse(output)



print(get_similar_movie("Deep Impact"))
print(get_similar_movie("Matrix"))