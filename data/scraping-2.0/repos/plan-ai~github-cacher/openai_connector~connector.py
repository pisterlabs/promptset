from pydantic import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI


# Define your desired data structure.
class IssueAnnotation(BaseModel):
    difficulty: str = Field(description="difficulty of issue")
    language: str = Field(description="programming language of issue")

    # You can add custom validation logic easily with Pydantic.
    @validator("difficulty")
    def test_difficult(cls, field):
        if field.lower() not in ["easy", "medium", "hard"]:
            raise ValueError("Invalid difficulty!")
        return field

    @validator("language")
    def test_language(cls, field):
        return field


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=IssueAnnotation)


class OpenAIConnector:
    def __init__(self, api_key):
        self.__api_key = api_key

    def annotate(self, issue_title, issue_description):
        model = OpenAI(
            model_name="text-davinci-003", temperature=0, openai_api_key=self.__api_key
        )
        prompt = PromptTemplate(
            template="Here is a github issue title and description, please annotate it.\n{format_instructions}\nTitle: {title}\nDescription: {description}",
            input_variables=["title", "description"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        input = prompt.format_prompt(title=issue_title, description=issue_description)

        output = model(input.to_string())

        return parser.parse(output)
