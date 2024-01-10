import os
from helper.token_helper import num_tokens_from_string
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain_pydantic
from pypdf import PdfReader
from extraction.schemas import Course
import backoff
# from helper.fake_data import fake_data
from pydantic import ValidationError


class Extraction:
    def __init__(self, openai_api_key):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(openai_api_key=openai_api_key,
                              temperature=0, model_name="gpt-3.5-turbo-1106")
        self.schema = Course
        self.extraction_chain = create_extraction_chain_pydantic(
            pydantic_schema=self.schema, llm=self.llm)

    def scrape(self, file):
        reader = PdfReader(file)
        number_of_pages = len(reader.pages)
        text = ''
        for i in range(number_of_pages):
            page = reader.pages[i]
            text += page.extract_text()
        return text

    @backoff.on_exception(backoff.expo, ValidationError, max_tries=3)
    def extract(self, text):
        result = self.extraction_chain.run(text)
        return result[0].dict()
