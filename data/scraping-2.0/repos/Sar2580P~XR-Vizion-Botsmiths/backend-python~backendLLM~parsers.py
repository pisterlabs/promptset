from pydantic import validator
from pydantic import BaseModel, Field
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

# create output parser class
class ArticleSummary(BaseModel):
  title: str = Field(description="Title of the article")
  summary: str = Field(description="Bulleted summary of the article")

  # validating whether the generated summary has at least 3 bullet points
  # allow_reuse : whether to track and raise an error if another validator refers to decorated function
  @validator('summary', allow_reuse=True)   
  def has_three_or_more_lines(cls, list_of_lines):
    if len(list_of_lines) < 2:
      raise ValueError("Generated summary has less than three bullet points!")
    return list_of_lines

# set up output parser
summary_parser = PydanticOutputParser(pydantic_object=ArticleSummary)

#_______________________________________________________________________________________________________________________

chat_format_template = '''
Break the text into bulleted points to highlight important information
Also, use bold and italics to highlight important keywords.

Text: 

{query}
'''

chat_format_prompt = PromptTemplate(
    input_variables=["query"],
    template=chat_format_template,
)
# retry_parser_chat = RetryOutputParser(parser=StrOutputParser(), retry_prompt=chat_format_prompt)