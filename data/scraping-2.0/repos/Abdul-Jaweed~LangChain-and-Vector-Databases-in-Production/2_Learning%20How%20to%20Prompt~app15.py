# OutputFixingParser

# This method tries to fix the parsing error by looking at the model’s response and the previous parser. It uses a Large Language Model (LLM) to solve the issue. We will use GPT-3 to be consistent with the rest of the lesson, but it is possible to pass any supported model. Let’s start by defining the Pydantic data schema and show a sample error that could occur.

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

parser = PydanticOutputParser(pydantic_object=Suggestions)

missformatted_output = '{"words": ["conduct", "manner"], "reasoning": ["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}'

parser.parse(missformatted_output)


from langchain.llms import OpenAI
from langchain.output_parsers import OutputFixingParser

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

model = OpenAI(
    openai_api_key=apikey,
    model_name='text-davinci-003',
    temperature=0.0
)


outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
outputfixing_parser.parse(missformatted_output)