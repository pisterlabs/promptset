from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from pydantic import BaseModel

from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown

line_template = '\t"{name}": {type}  // {description}'


class ResponseSchema(BaseModel):
    name: str
    description: str
    type: str = "string"


def _get_sub_string(schema: ResponseSchema) -> str:
    return line_template.format(
        name=schema.name, description=schema.description, type=schema.type
    )


class StructuredDictOutputParser(StructuredOutputParser):
    response_schemas: list[ResponseSchema]

    def get_format_instructions(self) -> str:
        schema_str = '{"response":[ ' + "\n".join(
            [_get_sub_string(schema) for schema in self.response_schemas]
        )+'],}'
        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    def parse(self, text: str):
        return parse_json_markdown(text)["response"]

    @property
    def _type(self) -> str:
        return "structured_dict"


def get_format_parser():
    question_text_schema = ResponseSchema(name="question_text",
                                          description="Question text")
    options_schema = ResponseSchema(name="options",
                                    description=" Options that are associated to questions",
                                    type="list[str]")
    correct_option_idx_schema = ResponseSchema(name="correct_option_idx",
                                               description="Correct option index starting from 0",
                                               type="int")
    explanation_schema = ResponseSchema(name="explanation",
                                        description="Some explanation related to the quiz")

    response_schemas = [question_text_schema, options_schema,
                        correct_option_idx_schema, explanation_schema]
    output_parser = StructuredDictOutputParser.from_response_schemas(
        response_schemas)

    return output_parser


prompt_template = """
Generate as many questions as possible for Multiple choice question quizzes for {topic}, be pricise and factual.
Make sure that the following informations are generated. Options should be clear and should have misconceptual options to confuse student.
One option should be correct in the options.

question_text: Question text here, character limit is strictly less than 255.

options: Get multiple options here and output them as a comma separated Python list. Each option must of less than 100 characters.

correct_option_idx: Get the index of the options which is correct. Index starts from zero and output them as python integer.

explanation: character limit is strictly less than 190. Explain or give factual information about the quiz.

{format_instructions}
"""
