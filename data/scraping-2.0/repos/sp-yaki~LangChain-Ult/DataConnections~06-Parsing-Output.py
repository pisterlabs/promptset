from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.output_parsers import (
    CommaSeparatedListOutputParser,
    DatetimeOutputParser,
    OutputFixingParser
)

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

model = ChatOpenAI(temperature=0)

# output_parser = CommaSeparatedListOutputParser()
# format_instructions = output_parser.get_format_instructions()

# human_template = '{request} {format_instructions}'
# human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
# chat_prompt.format_prompt(request="give me 5 characteristics of dogs",
#                    format_instructions = output_parser.get_format_instructions())

# request = chat_prompt.format_prompt(request="give me 5 characteristics of dogs",
#                    format_instructions = output_parser.get_format_instructions()).to_messages()

# result = model(request)
# print(output_parser.parse(result.content))

# //////////////////////////////
# output_parser = DatetimeOutputParser()
# # print(output_parser.get_format_instructions())

# template_text = "{request}\n{format_instructions}"
# human_prompt=HumanMessagePromptTemplate.from_template(template_text)

# chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

# # print(chat_prompt.format(request="When was the 13th Amendment ratified in the US?",
# #                    format_instructions=output_parser.get_format_instructions()
# #                    ))

# request = chat_prompt.format_prompt(request="What date was the 13th Amendment ratified in the US?",
#                    format_instructions=output_parser.get_format_instructions()
#                    ).to_messages()

# result = model(request,temperature=0)
# misformatted = result.content
# # print(misformatted)
# new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=model)
# print(new_parser.parse(misformatted))

# //////////////////////////////
# output_parser = DatetimeOutputParser()
# system_prompt = SystemMessagePromptTemplate.from_template("You always reply to questions only in datetime patterns.")
# template_text = "{request}\n{format_instructions}"
# human_prompt=HumanMessagePromptTemplate.from_template(template_text)

# chat_prompt = ChatPromptTemplate.from_messages([system_prompt,human_prompt])

# request = chat_prompt.format_prompt(request="What date was the 13th Amendment ratified in the US?",
#                    format_instructions=output_parser.get_format_instructions()
#                    ).to_messages()

# result = model(request,temperature=0)
# print(output_parser.parse(result.content))

# //////////////////////////////
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Scientist(BaseModel):
    name: str = Field(description="Name of a Scientist")
    discoveries: list = Field(description="Python list of discoveries")
query = 'Name a famous scientist and a list of their discoveries' 
parser = PydanticOutputParser(pydantic_object=Scientist)
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
_input = prompt.format_prompt(query="Tell me about a famous scientist")
output = model(_input.to_messages())
print(parser.parse(output))
print(output)