"""Provides user search processing and AWS Bedrock language model calling functionality."""
import json
from json.decoder import JSONDecodeError
from typing import Any

import boto3
from langchain.llms import Bedrock
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.exceptions import LangChainException
from langchain.schema.messages import SystemMessage

from llm_api.config import BedrockModel, Settings


class BedrockModelCallError(Exception):
    """Generate a custom exception for Bedrock API errors."""


class BedrockCaller:
    """Process prompts and call Bedrock LLMs."""

    def __init__(self, settings: Settings) -> None:
        """
        Class constructor.

        Args:
            settings (Settings): Pydantic settings object.
        """
        self.settings = settings
        self.boto3_client = self.get_boto3_client()
        self.client = self.get_client(self.settings.aws_bedrock_model_id)

    def get_boto3_client(self) -> Any:  # noqa: ANN401
        """
        Retrieve LangChain client to call Bedrock models.

        Returns
            Bedrock: Langchain Bedrock client object
        """
        return boto3.client(
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key.get_secret_value(),
            region_name="us-east-1",
            service_name="bedrock-runtime",
        )

    def get_client(self, bedrock_model_id: BedrockModel) -> Bedrock:
        """
        Retrieve LangChain client to call Bedrock models.

        Returns
            Bedrock: Langchain Bedrock client object
        """
        return Bedrock(
            client=self.boto3_client,
            model_id=bedrock_model_id,
            model_kwargs={
                "max_tokens_to_sample": 4096,
                "temperature": 0.5,
                "top_k": 250,
                "top_p": 1,
                "stop_sequences": ["\n\nHuman:"],
            },
        )

    @staticmethod
    def generate_prompt() -> ChatPromptTemplate:
        """
        Generate a prompt from user input to send to models.

        The prompt includes `system` and `user` roles to define expected
        input and output formats.

        Args:
            user_input (str): User search input.

        Returns:
            ChatPromptTemplate: A list of dictionaries containing roles and content.
        """
        user_template = "{text}"
        return ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=" ".join(
                        "You are a helpful search assistant that extracts \
                    useful entities and relationships from user search queries \
                    and returns the answer as JSON. The user can only understand \
                    JSON responses, and cannot understand any free text outside \
                    of a JSON-formatted response. Users are inquisitive learners \
                    who would like to find out about a particular subject. \
                    Your job is to aid them in this task by providing them \
                    with information.".split()
                    ),
                ),
                SystemMessage(
                    content=" ".join(
                        "Users will provide you with a search, as a string. \
                    You should examine this string and determine any entities directly \
                    found in the string in addition to as many other relevant connected entities \
                    not directly found in the user search. \
                    You should provide the user with as many relevant entities as possible \
                    while keeping entities specific to the search. Provide a minimum of 5 entities.\
                    Please also provide a paragraph of 3-5 sentences describing the connections \
                    between the entities, keeping this information specific and \
                    relevant to the user search.".split()
                    ),
                ),
                SystemMessage(
                    content=" ".join(
                        "Respond to the user by providing valid JSON in the format \
                        specified in the example shown here. Do not include any text outside \
                        of the JSON response. Format your JSON response by \
                        matching the following example, which is shown \
                        between the two sets of three backtick characters (`). Dictionary keys \
                        should be taken literally, \
                        but dictionary values are indicative".split()
                    ),
                ),
                SystemMessage(
                    content="```\
                    {{'entities': [ \
                        {{'uri': 'entity name', 'description': 'entity description', \
                        'wikipedia_url': 'entity wikipedia url'}} \
                    ]}},\
                    'connections': [\
                    {{'from': 'uri'\
                    'to': 'uri'\
                    'description': 'paragraph describing entity-entity relationship'\
                    }}]\
                    }} \
            ```",
                ),
                HumanMessagePromptTemplate.from_template(user_template),
            ]
        )

    async def call_model(
        self,
        prompt_template: ChatPromptTemplate,
        user_search: str,
        alternative_model: BedrockModel | None = None,
    ) -> dict[str, str]:
        """
        Call the external Bedrock model specified with a defined prompt via LangChain.

        Args:
            prompt_template (ChatPromptTemplate): LangChain ChatPromptTemplate
                containing system instructions and any example formatting required.
            user_search (str): User's search as a string.
            alternative_model: Alternative model to use if not using the default model.

        Raises:
            BedrockModelCallError: Index error due to unexpected response format
            BedrockModelCallError: JSONDecode error due to unexpected response format
            BedrockModelCallError: General LangChain exception


        Returns:
            dict[str, str]: Model JSON response as a dictionary.
        """
        try:
            if alternative_model:
                self.client = self.get_client(alternative_model)

            self.chain = prompt_template | self.client
            model_response = await self.chain.ainvoke({"text": user_search})

            try:
                return json.loads(model_response.split("```")[1].strip("json"))
            except IndexError as unexpected_response_error:
                message = f"Unable to parse model output as expected. {unexpected_response_error}"
                raise BedrockModelCallError(message) from unexpected_response_error
            except JSONDecodeError as json_error:
                message = f"Error decoding model output. {json_error}"
                raise BedrockModelCallError(message) from json_error
        except ValueError as bedrock_model_call_error:
            message = f"Error calling model. {bedrock_model_call_error}"
            raise BedrockModelCallError(message) from bedrock_model_call_error
        except LangChainException as langchain_error:
            message = f"Error sending prompt to LLM. {langchain_error}"
            raise BedrockModelCallError(message) from langchain_error
