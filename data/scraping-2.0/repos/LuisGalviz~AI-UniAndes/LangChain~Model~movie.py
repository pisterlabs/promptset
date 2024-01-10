from langchain.pydantic_v1 import BaseModel, Field, validator
from typing import List

# Here's another example, but with a compound typed field.
class Movie(BaseModel):
    actor: str = Field(description="name of an actor")
    film: str = Field(description="name of film")
    name: str = Field(description="name of movie")
    genre: str = Field(description="type genre of movie")
    movies: List[str] = Field(description="list of names of movies they starred in")