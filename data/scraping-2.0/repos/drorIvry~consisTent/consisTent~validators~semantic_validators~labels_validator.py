import json
from typing import List
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

from ..base_validator import Validator


class LabelsValidator(Validator):
    def __init__(
        self,
        openai_key: str,
    ):
        self._model = ChatOpenAI(
            temperature=0,
            openai_api_key=openai_key,
            max_tokens=1000,
        )

        response_schemas = [
            ResponseSchema(
                name="label",
                description="This is the input_label from the user",
            ),
            ResponseSchema(
                name="passed",
                description="This is the validation boolean result of the input against this labels. values are true / false",  # noqa E501
            ),
            ResponseSchema(
                name="match_score",
                description="A score 0-100 of how close you think the match is between user input and your match",  # noqa E501
            ),
            ResponseSchema(
                name="reason",
                description="Why did the input failed / passed this labels",
            ),
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        self._template = """
        you are an input validator.
        given a set of labels your goal is to decide
        if the input matches the given labels.

        {format_instructions}

        Wrap your final output with closed and open brackets (a list of json objects)

        labels:
        {labels}

        INPUT:
        {user_input}

        YOUR RESPONSE:
            """

        self._prompt = ChatPromptTemplate(
            messages=[HumanMessagePromptTemplate.from_template(self._template)],
            input_variables=["labels", "user_input"],
            partial_variables={"format_instructions": format_instructions},
        )

    def validate(
        self,
        labels: List[str],
        model_output: str,
    ):
        parsed_labels = ", ".join(labels)

        _input = self._prompt.format_prompt(
            labels=parsed_labels,
            user_input=model_output,
        )

        output = self._model(_input.to_messages())

        structured_data = json.loads(output.content)

        for entry in structured_data:
            assert (
                entry["passed"] is True
            ), f"llm validation check validation failed on {entry['label']} for {entry['reason']}"  # noqa: E501
