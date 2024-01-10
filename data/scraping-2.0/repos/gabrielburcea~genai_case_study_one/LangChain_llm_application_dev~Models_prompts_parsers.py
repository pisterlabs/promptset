# Databricks notebook source
"""LangChain: Models, Prompts and Output Parsers¶
Outline
Direct API calls to OpenAI
API calls through LangChain:
Prompts
Models
Output parsers"""



# COMMAND ----------

import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

"""Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video."""

# COMMAND ----------

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

# COMMAND ----------

"""Chat API : OpenAI¶
Let's start with a direct API calls to OpenAI."""

def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]


# COMMAND ----------

get_completion("What is 1+1?")

# COMMAND ----------

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

# COMMAND ----------

style = """American English \
in a calm and respectful tone
"""

# COMMAND ----------

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print(prompt)

# COMMAND ----------

response = get_completion(prompt)

# COMMAND ----------

response

# COMMAND ----------

"""Chat API : LangChain
Let's try how we can do the same using LangChain."""

# COMMAND ----------

"""Model"""

from langchain.chat_models import ChatOpenAI
# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0, model=llm_model)
chat

# COMMAND ----------

"""Prompt template"""
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""


# COMMAND ----------

from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(template_string)


# COMMAND ----------

prompt_template.messages[0].prompt

# COMMAND ----------

prompt_template.messages[0].prompt.input_variables

# COMMAND ----------

customer_style = """American English \
in a calm and respectful tone
"""

# COMMAND ----------

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

# COMMAND ----------

customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

# COMMAND ----------

print(type(customer_messages))
print(type(customer_messages[0]))

# COMMAND ----------

print(customer_messages[0])

# COMMAND ----------

# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)

# COMMAND ----------

print(customer_response.content)

# COMMAND ----------

service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

# COMMAND ----------

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""

# COMMAND ----------

service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply)

print(service_messages[0].content)

# COMMAND ----------

service_response = chat(service_messages)
print(service_response.content)

# COMMAND ----------

"""Output Parsers
Let's start with defining how we would like the LLM output to look like:"""

# COMMAND ----------

{
  "gift": False,
  "delivery_days": 5,
  "price_value": "pretty affordable!"
}

# COMMAND ----------

customer_review = """\
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

# COMMAND ----------

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

# COMMAND ----------

from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(review_template)
print(prompt_template)

# COMMAND ----------

messages = prompt_template.format_messages(text=customer_review)
chat = ChatOpenAI(temperature=0.0, model=llm_model)
response = chat(messages)
print(response.content)

# COMMAND ----------

type(response.content)

# COMMAND ----------

# You will get an error by running this line of code 
# because'gift' is not a dictionary
# 'gift' is a string
response.content.get('gift')

# COMMAND ----------

"""Parse the LLM output string into a Python dictionary"""

# COMMAND ----------

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

# COMMAND ----------

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

# COMMAND ----------

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# COMMAND ----------

format_instructions = output_parser.get_format_instructions()

# COMMAND ----------

print(format_instructions)

# COMMAND ----------

review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

messages = prompt.format_messages(text=customer_review, 
                                format_instructions=format_instructions)

# COMMAND ----------

print(messages[0].content)

# COMMAND ----------

response = chat(messages)

# COMMAND ----------

print(response.content)

# COMMAND ----------

output_dict = output_parser.parse(response.content)

# COMMAND ----------

output_dict

# COMMAND ----------

type(output_dict)

# COMMAND ----------

output_dict.get('delivery_days')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


