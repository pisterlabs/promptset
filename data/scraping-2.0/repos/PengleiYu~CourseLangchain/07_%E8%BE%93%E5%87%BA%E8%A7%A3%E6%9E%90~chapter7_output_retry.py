template = """
Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:
"""

from pydantic import BaseModel, Field


class Action(BaseModel):
    action: str = Field(description='action to take')
    action_input: str = Field(description='input to the action')


from langchain.output_parsers.pydantic import PydanticOutputParser

# 注意这里的pydantic必须是v1版本，否则抛出的异常不同，会导致OutputFixingParser无法捕获异常
parser = PydanticOutputParser(pydantic_object=Action)

from langchain.prompts.prompt import PromptTemplate

prompt_template = PromptTemplate.from_template(template=template, partial_variables={
    'format_instructions': parser.get_format_instructions()})

prompt_value = prompt_template.format_prompt(query='What are the colors of Orchid?')
print('prompt value=', prompt_value)

bad_response = '{"action": "search"}'
print('base_response', bad_response)
# parser.parse(bad_response)

from langchain.chat_models.openai import ChatOpenAI
from langchain.output_parsers.fix import OutputFixingParser

output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
fixed_response = output_fixing_parser.parse(bad_response)
print('fixed_response', fixed_response)

from langchain.output_parsers.retry import RetryWithErrorOutputParser

retry_output_parser = RetryWithErrorOutputParser.from_llm(llm=ChatOpenAI(temperature=0), parser=parser)
result = retry_output_parser.parse_with_prompt(completion=bad_response, prompt_value=prompt_value)
print('retryWithError', result)
