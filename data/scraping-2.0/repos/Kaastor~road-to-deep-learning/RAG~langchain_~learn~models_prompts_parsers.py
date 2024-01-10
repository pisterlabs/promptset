import os
import openai

from dotenv import load_dotenv, find_dotenv

from langchain_.learn.consts import OPENAI_API_KEY

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = OPENAI_API_KEY

'''
CHAT API: OPENAI
'''


# Asks OpenAI for answer and returns it
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


get_completion("What is 1+1?")
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
style = """Polish \
in a calm and respectful tone
"""
prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print(prompt)
#response = get_completion(prompt)
#print(response)

'''
CHAT API: LANGCHAIN
'''
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.0)

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)
print(prompt_template.messages[0].prompt)
print(prompt_template.messages[0].prompt.input_variables)

customer_style = """Polish \
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

# Fill prompt template with input variables
customer_messages = prompt_template.format_messages(
    style=customer_style,
    text=customer_email)
print(type(customer_messages))
print(type(customer_messages[0]))
print(customer_messages[0])

# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)
print(customer_response.content)

'''
OUTPUT PARSERS
'''