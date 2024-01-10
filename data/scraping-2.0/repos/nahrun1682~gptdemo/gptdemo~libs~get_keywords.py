import openai
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from . import config

def return_keywords_for_google(transcript):
    
    #コンマ区切りのリストを返す
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
            template=config.template_for_google_search,
            input_variables=["input"],
            partial_variables={"format_instructions": format_instructions},
    )

    model = OpenAI(temperature=0)
    #llm = OpenAI(model_name=config.model_engine)

    _input = prompt.format(input=transcript)
    output = model(_input)
    return output_parser.parse(output)