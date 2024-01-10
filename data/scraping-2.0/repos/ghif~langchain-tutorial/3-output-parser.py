from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


"""
Load OpenAI API key 
"""
_ = load_dotenv(find_dotenv())

"""
Product review analysis
"""

user_review = """
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""


# Create response schema
sentiment_schema = ResponseSchema(
    name="sentiment",
    description="""
    Is the review positive or negative? \
    Answer "positive" or "negative".
    """
)

gift_schema = ResponseSchema(
    name="gift",
    description="""
    Was the product purchased as a gift for someone else?
    Answer True if yes, False if no or unknown.
    """
)

# Create structured output parser
output_parser = StructuredOutputParser.from_response_schemas(
    [sentiment_schema, gift_schema]
)

format_instructions = output_parser.get_format_instructions()
# print(format_instructions)

template = """
From the following text, extract the following information:

sentiment: Is the review positive or negative? \
Answer "positive" or "negative".

gift: Was the product purchased as a gift for someone else?
Answer True if yes, False if no or unknown.

text: {review}

{format_instructions}
"""

# Create model
chat_llm = ChatOpenAI(temperature=0.0)


# Design prompt
prompt = ChatPromptTemplate.from_template(template)

final_prompt = prompt.format(
    review=user_review,
    format_instructions=format_instructions
)

# Inference
response = chat_llm.predict(final_prompt)

print(f"Final prompt: {final_prompt}")
print(f"Response: {response}")