from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.output_parsers import (
    OutputFixingParser,
    PydanticOutputParser,
    RetryWithErrorOutputParser,
)
from langchain.prompts import (
    PromptTemplate,
)
from pydantic import BaseModel, Field
from util import initialize

initialize()

template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:"""


class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")


parser = PydanticOutputParser(pydantic_object=Action)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
prompt_value = prompt.format_prompt(query="who is leo di caprios gf?")

bad_response = '{"action": "search"}'
# parser.parse(bad_response)

# OutputFixingParserでは、action_inputが必要なことは分かっても、action_inputに入れるべき値が分からない
fix_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
fix_parser_result = fix_parser.parse(bad_response)
print(fix_parser_result)

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=parser, llm=OpenAI(temperature=0)
)
retry_parser_result = retry_parser.parse_with_prompt(bad_response, prompt_value)
print(retry_parser_result)
