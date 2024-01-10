import json
import logging
from typing import Optional, List, Union, Dict
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate, AIMessagePromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from pydantic import BaseModel, Field, ConfigDict

from llm_prompting_gen.constants import INSTRUCTOR_USER_NAME

class FewShotHumanAIExample(BaseModel):
    human: Optional[str] = Field(None, description="Example human message.", examples=["What is 2 + 3?"])
    ai: str = Field(description="Example ai message as answer to the human message. Or standalone example without human message.", examples=["5"])

class PromptElements(BaseModel):
    """
    Defines all elements of a prompt
    """
    role: Optional[str] = Field(None, description="The role in which the LLM should respond")
    instruction: Optional[str] = Field(None, description="The task of the LLM")
    context: Optional[str] = Field(None, description="Context with relevant information to solve the task")
    output_format: Optional[str] = Field(None, description="Description how the LLM output format should look like")
    examples: Optional[Union[List[FewShotHumanAIExample], List[str]]] = Field(
        None, description="Few shot examples. Can be either human ai interactions or a list of examples")
    input: Optional[str] = Field(
        None, description="Target which the LLM should execute the task on. Could be for example a user question, or a text block to summarize.")

    model_config = ConfigDict(
        extra='allow',
    )


    def is_any_set(self) -> bool:
        """
        Whether at least one prompt element is set.
        Otherwise, an LLM cannot handle prompt input and therefore False is returned
        """
        return any([self.role, self.instruction, self.context, self.examples, self.input, self.output_format])

    def get_example_system_prompt_template(self) -> SystemMessagePromptTemplate:
        """Transforms list of examples into SystemMessagePromptTemplate
            "Example <n>:" is appended as prefix of each example

            Example:
                self.examples = ["positive", "negative", "neutral"]
                t = self.get_example_system_prompt_template()
                # t equals (pseudo representation of SystemMessagePromptTemplate)
                SystemMessagePromptTemplate('''
                    Example 1: positive
                    Example 2: negative
                    Example 3: neutral
                    ''')
        """
        assert self.examples, "examples are not set yet"
        assert type(self.examples) == list and type(self.examples[0]) == str, "examples is not a list of strings"

        prompt_template = ""
        for i, example in enumerate(self.examples):
            prompt_template += f"\nExample {i+1}: {example}"
        return SystemMessagePromptTemplate.from_template(prompt_template, additional_kwargs={"name": INSTRUCTOR_USER_NAME})

    def get_example_msg_prompt_templates(self) -> List[Union[SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate]]:
        """
        Returns a list of message prompt template depending on the example format
        If human_ai_interaction is set:
            Returns list of langchain human and ai prompt templates
        If examples are set and no human_ai_interaction:
            Returns list of single langchain system message prompt template
        """
        assert self.examples, "examples are not set yet"
        # Return SystemMessagePromptTemplate in case only simple string examples are set
        if self.examples and not type(self.examples[0]) == FewShotHumanAIExample:
            return [self.get_example_system_prompt_template()]
        assert type(self.examples[0]) == FewShotHumanAIExample, "human ai interaction examples are not set yet"

        messages = []
        for example in self.examples:
            messages.append(HumanMessagePromptTemplate.from_template(example.human))
            messages.append(AIMessagePromptTemplate.from_template(example.ai))
        return messages


class PromptEngineeringMessages:
    """
    Dataclass which contains all messages to create a LLM Output
    based on prompt engineering techniques.
    Transforms strings into langchain prompt template messages.
    """

    def __init__(self, messages: Dict[str, Union[BaseMessagePromptTemplate, List[BaseMessagePromptTemplate]]]):
        """
        Note: messages can contain custom keys and do not require but can match the PromptElements fields.
        Examples are a special case, where we have multiple BaseMessagePromptTemplate for one key.

        :param messages: Look up dict matching prompt element key with corresponding PromptTemplate
        """
        self.messages = messages


    @classmethod
    def from_pydantic(cls, pe_elements: PromptElements):
        """Init class by PromptElements pydantic"""
        messages = {}
        if pe_elements.role:
            messages["role"] = SystemMessagePromptTemplate.from_template(pe_elements.role, additional_kwargs={"name": INSTRUCTOR_USER_NAME})
        if pe_elements.instruction:
            messages["instruction"] = SystemMessagePromptTemplate.from_template(
                pe_elements.instruction, additional_kwargs={"name": INSTRUCTOR_USER_NAME})
        if pe_elements.context:
            messages["context"] = SystemMessagePromptTemplate.from_template(pe_elements.context, additional_kwargs={"name": INSTRUCTOR_USER_NAME})
        if pe_elements.output_format:
            # Create output format system message
            output_format_prompt = PromptTemplate(
                template="{format_instructions}",
                input_variables=[],
                partial_variables={"format_instructions": pe_elements.output_format}
            )
            messages["output_format"] = SystemMessagePromptTemplate(prompt=output_format_prompt, additional_kwargs={"name": INSTRUCTOR_USER_NAME})
        if pe_elements.examples:
            messages["examples"] = pe_elements.get_example_msg_prompt_templates()
        if pe_elements.input:
            messages["input"] = HumanMessagePromptTemplate.from_template(pe_elements.input)

        # Add extra fields as system messages
        for extra_field, extra_value in pe_elements.model_extra.items():
            messages[extra_field] = SystemMessagePromptTemplate.from_template(extra_value, additional_kwargs={"name": INSTRUCTOR_USER_NAME})

        return cls(messages=messages)

    @classmethod
    def from_json(cls, file_path):
        """Init class by json file"""
        with open(file_path, "r") as fp:
            data_dict = json.load(fp)
        return cls.from_pydantic(PromptElements(**data_dict))

    @classmethod
    def from_yaml(cls, file_path):
        # TODO
        raise NotImplementedError

    def get_flattened_messages(self, message_order: Optional[List[str]] = None) -> List[BaseMessagePromptTemplate]:
        """Returns list of valid messages. Examples are  flattened in order to get one dimension output.
            Optionally a message order can be provided to define the output order
        """
        flattened_messages = []

        message_keys = message_order or list(self.messages.keys())
        for message_key in message_keys:
            if message_key not in self.messages:
                logging.warning(
                    f"messages do not contain key '{message_key}'")
                continue
            message_or_list = self.messages[message_key]
            # Case message is valid
            if isinstance(message_or_list, BaseMessagePromptTemplate):
                flattened_messages.append(message_or_list)
            # Case list of messages
            elif isinstance(message_or_list, list):
                for message in message_or_list:
                    # check if message is valid
                    if isinstance(message, BaseMessagePromptTemplate):
                        flattened_messages.append(message)

        return flattened_messages

    def get_chat_prompt_template(self, message_order: Optional[List[str]] = None) -> ChatPromptTemplate:
        """Combines all prompt element messages into one chat prompt template

        :param message_order: Optional list of prompt element keys, to define message order of final chat prompt template
        :return:
        """

        if message_order:
            # Print warning in case message_order does not include all fields available
            excluded_fields = set(self.messages.keys()) - set(message_order)
            if excluded_fields:
                logging.warning(f"'message_order' does not include fields {excluded_fields}. They will be ignored for chat prompt template creation")

        return ChatPromptTemplate.from_messages(self.get_flattened_messages(message_order))