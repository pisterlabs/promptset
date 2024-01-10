from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


class LocationChain(LLMChain):
    def __init__(self):
        location_template = """What is the location of the weather request? Answer in the following format: city, state. If no location is present in the weather request or chat history, answer Denver, CO.

chat history:
{history}

weather request: {input}

Location:"""

        super().__init__(
            llm=OpenAI(model="gpt-3.5-turbo-instruct"),
            prompt=PromptTemplate.from_template(location_template),
            output_key="location",
        )
