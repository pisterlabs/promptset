from typing import List

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.llms import OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from app.golems.models.knowledge_elements.knowledge_elements import KnowledgeElements

llm = OpenAIChat(model_name='gpt-3.5-turbo')

ke_parser = PydanticOutputParser(pydantic_object=KnowledgeElements)
prompt = PromptTemplate(
    input_variables = ["knowledge_elements"],
    template = '''
    Here is a list of different knowledge_elements:
    {knowledge_elements}
    format like this:{format_instructions}
    ''',
    partial_variables={"format_instructions": ke_parser.get_format_instructions()}
)
list_generating_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
knowledge_element_parser = OutputFixingParser.from_llm(llm = llm, parser = ke_parser)

def parse_elements(elements_string: str) -> List[str]:
    raw_response = list_generating_chain.predict(
        knowledge_elements = elements_string
    ) 

    parsed_response = knowledge_element_parser.parse(raw_response)

    return parsed_response.elements