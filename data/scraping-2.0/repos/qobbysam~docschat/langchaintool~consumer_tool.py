

from .config import DefaultConfig
from typing import List
from pydantic import BaseModel, Field, validator 
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser


prompt_template = "provide the title and a list of suggested keywords that will be saved in a database from this page of text. \n{format_instructions} \n{text}"


class TitleKeyword(BaseModel):
    title: str = Field(description="title of pdf")
    keywords: List[str] = Field(description="list of suggested keywords")

class ConsumptionLLM:
    
    def __init__(self, config=DefaultConfig()) -> None:
        self.config = config
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            openai_api_key=self.config.api_key)
        self.parser = PydanticOutputParser(pydantic_object=TitleKeyword)

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )



    def get_title_keywords(self, page_text):


        chain = LLMChain(prompt=self.prompt, llm=self.llm)

        output = chain.run(text=page_text)

        parsed = self.parser.parse(output)

        title = parsed.title

        keyword = parsed.keywords

        return title,keyword