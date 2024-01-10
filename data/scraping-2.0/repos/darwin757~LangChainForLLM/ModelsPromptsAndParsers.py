import os
import openai
import datetime

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

_ = load_dotenv(find_dotenv())  # take environment variables from .env.

openai.api_key = os.getenv("OPENAI_API_KEY")

# account for deprecation of LLM model
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

if current_date >= target_date:
    llm_model = ("gpt-3.5-turbo")
else:
    llm_model = ("gpt-3.5-turbo-0301")

def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print("Prompting AI")
print(prompt)

response = get_completion(prompt)
print("Response from AI")
print(response)
print("\n")

# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0, model=llm_model)
print("Printing the Chat as an Object")
print(chat)
print("\n")

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

print("Printing the Prompt from the Prompt Template")
print(prompt_template.messages[0].prompt)
print("\n")

print("Printing the Prompt's input_variables from the Prompt Template")
print(prompt_template.messages[0].prompt.input_variables)
print("\n")

customer_style = """American English \
in a calm and respectful tone
"""

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

print("Printing the Customer Messages type")
print(type(customer_messages))
print(type(customer_messages[0]))
print("Printing the first element of the Customer Messages list")
print(customer_messages[0])
# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)
print("Printing the Customer Response's Content")
print(customer_response.content)
print("\n")

service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""
service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply)

print("Printing the Service Messages Prompt")
print(service_messages[0].content)

print("Printing the Service Messages Reply")
service_response = chat(service_messages)
print(service_response.content)
print("\n")

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

prompt_template = ChatPromptTemplate.from_template(review_template)
print("Printing the Prompt Template")
print(prompt_template)

messages = prompt_template.format_messages(text=customer_review)
chat = ChatOpenAI(temperature=0.0, model=llm_model)
response = chat(messages)
print("Printing the Response Content")
print(response.content)
print("Printing the Response Type")
print(type(response.content))
print("\n")

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

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print("Printing the Format Instructions from the Structured Output Parser")
print(format_instructions)
print("\n")

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

print("Printing the Messages Prompt")
print(messages[0].content)
print("\n")

response = chat(messages)
print("Printing the Response Content")
print(response.content)
print("\n")

output_dict = output_parser.parse(response.content)
print("Printing the Output Dict")
print(output_dict)
print("\n")
print("Printing the Output Dict Type")
print(type(output_dict))
print("\n")

print("Printing the Delivery Days")
print(output_dict.get('delivery_days'))