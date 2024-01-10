# %%
import json
from typing import List
from langchain import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chat_models import ChatOpenAI

from ..base_validator import Validator


class RequestResponseValidator(Validator):
    def __init__(
        self,
        openai_key: str,
    ):
        self._model = ChatOpenAI(
            temperature=0.7,
            openai_api_key=openai_key,
        )

        response_schemas = [
            ResponseSchema(
                name="claim",
                description="This is the claim created",
            ),
            ResponseSchema(
                name="contradiction",
                description="This is the contradiction boolean result of the content against this claim. values are true / false",  # noqa E501
            ),
            ResponseSchema(
                name="match_score",
                description="A score 0-100 of how bad you think the contradiction is between content and your claim",  # noqa E501
            ),
            ResponseSchema(
                name="reason",
                description="Why did the claim failed / passed this content",
            ),
            ResponseSchema(
                name="included_in_content",
                description="Is the Claim included in the given content. values are true / false ",  # noqa E501
            ),
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        self._template = """
            Given a statement list all of the claims in the statement and
            assert of the given content contridicts any of the claims:

            {format_instructions}

            Wrap your final output with closed and open brackets
            (a list of json objects)

            STATEMENT
            {statement}

            CONTENT
            {content}

            answer only with the JSON
            YOUR RESPONSE:
            """

        self._prompt = PromptTemplate(
            template=self._template,
            input_variables=["statement", "content"],
            partial_variables={"format_instructions": format_instructions},
        )

    def validate(
        self,
        model_input: List[str],
        model_output: str,
    ):
        _input = self._prompt.format_prompt(
            statement=model_input,
            content=model_output,
        )

        output = self._model(_input.to_messages())

        print(output)
        structured_data = json.loads(output.content)
        print(structured_data)

        omitted_facts_count = 0
        for entry in structured_data:
            assert (
                entry["contradiction"] is False
            ), f"llm validation check validation failed on {entry['label']} for {entry['reason']}"  # noqa: E501
        if entry["included_in_content"] is False:
            omitted_facts_count += 1

        assert omitted_facts_count < len(
            structured_data
        ), "most of the input claims were ignored"
