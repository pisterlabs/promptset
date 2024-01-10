# To extract more concise and formatted code mappings from API document
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

response_schemas = [
    ResponseSchema(
        name="weather type",
        type="json",
        description="significant weather code mappings e.g. 0 maps to 'Clear night'",
    ),
    ResponseSchema(
        name="visibility",
        type="json",
        description="visibility code mappings e.g. VP maps to 'Very poor - Less than 1 km'",
    ),
    ResponseSchema(
        name="uv",
        type="json",
        description="UV index mappings e.g. 1-2 maps to 'Low exposure. No protection required. You can safely stay outside.'",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

template = """
Data should be extracted from the following:
---------
{context}
---------
{format_instructions}
"""

question = """
Extract meaningful labels against the codes for all of the following, including all codes for each:
1. Significant weather"
2. UV"
3. Visibility"
"""

chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(template),
    ],
    partial_variables={"format_instructions": format_instructions},
    input_variables=["context"],
)


def get_prompt():
    return chat_prompt
