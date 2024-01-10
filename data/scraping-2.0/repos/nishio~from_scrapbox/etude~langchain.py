import dotenv
import os
import openai
import langchain
from langchain.chat_models import ChatOpenAI

dotenv.load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

llm_model = "gpt-3.5-turbo"
chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

template_string = """\
Template into English:
{japanese}
"""

from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(template_string)

customer_messages = prompt_template.format_messages(japanese="こんにちは")

# Call the LLM to translate to the style of the customer message
#customer_response = chat(customer_messages)

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]


from langchain.agents import tool
from datetime import datetime
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

@tool
def get_now(text: str) -> str:
    """Returns current date and time, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return <Year>-<Month>-<Day> <Hour>:<Minute>:<Second>.<Fraction> \
    date - any date mathmatics should occur \
    outside this function."""
    return str(datetime.now())

llm = ChatOpenAI(temperature=0, model=llm_model)
agent = initialize_agent(
    [get_now], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)