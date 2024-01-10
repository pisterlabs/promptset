from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import (
    PromptTemplate,
)
from utils import AzureModel
from langchain.schema import HumanMessage


class CharacterPrompt:
    def __init__(self):
        self.response_schemas = [
            ResponseSchema(
                name="answer", description="answer to the user's question as asked"
            ),
            ResponseSchema(
                name="series",
                description="this gives the series from which you need to get all information",
            ),
            ResponseSchema(
                name="character",
                description="this is the character of the series you have to behave like",
            ),
        ]

        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.response_schemas
        )

        self.format_instructions = self.output_parser.get_format_instructions()

    def create_prompt(
        self, series, character, question, serials, comics, stories, events
    ):
        # creates the prompt with details
        prompt = PromptTemplate(
            template="""Answer the user question as best as possible and try to be frank and give long answers as possible. Make conversation interactive. I want you to act like {character} from {series}. I want you to respond and answer like {character} using the tone, manner and vocabulary {character} would use. Do not write any explanations. Only answer like {character}.
             You must know all of the knowledge of {character}.\n
             latest details about characters: \n
             serials the character was in: {serials},\n
             comics of the character: {comics},\n
             stories of the character: {stories},\n
             events of the character: {events},\n
             format instructions:{format_instructions}\n
             question: {question}""",
            input_variables=[
                "question",
                "series",
                "character",
                "serials",
                "comics",
                "stories",
                "events",
            ],
            partial_variables={"format_instructions": self.format_instructions},
        )
        _input = prompt.format_prompt(
            question=question,
            series=series,
            character=character,
            serials=serials,
            comics=comics,
            stories=stories,
            events=events,
        )
        return _input

    def get_response_from_llm(
        self, series, character, question, serials, comics, stories, events
    ):
        # get response from llm
        _input = self.create_prompt(
            series, character, question, serials, comics, stories, events
        )

        llm_object = AzureModel()
        llm = llm_object.get_llm_model()
        response = llm([HumanMessage(content=_input.to_string())]).content
        parsed_response = self.output_parser.parse(response)
        return parsed_response
