from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.llms import OpenAI
from util.log import logger
from langchain.pydantic_v1 import BaseModel, Field
from datetime import datetime


class BasePrompt:
    def __init__(
        self,
        output_schema,
        base_templete: str = "{query}\n\n{format_instructions}\n",
        model_name: str = "gpt-3.5-turbo",
    ):
        self._output_schema = output_schema
        self._base_templete = base_templete
        self._model = OpenAI(model_name=model_name)

    def gen_prompt(self, query):
        parser = PydanticOutputParser(pydantic_object=self._output_schema)

        base_prompt = PromptTemplate(
            template=self._base_templete,
            input_variables=["query"],
            validate_template=True,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        self.prompt = base_prompt.format_prompt(query=query)

    def get_prompt(self):
        return self.prompt

    def run_prompt(self):
        prompt = self.prompt.to_string()
        logger.info("Prompt\n%s", prompt)
        return self._model(prompt)


class OutputSchema(BaseModel):
    ad_type: str = Field(description="広告の種類")
    change_start_date: datetime = Field(description="変更が始まる時期")
    content: str = Field(description="変更内容について")
