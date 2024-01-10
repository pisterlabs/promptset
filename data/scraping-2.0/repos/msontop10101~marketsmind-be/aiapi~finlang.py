import os
os.environ["OPENAI_API_KEY"] = "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"

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.llms import OpenAI
llm = OpenAI()  # Replace 'openai_api_key' with your actual OpenAI API key

# Define the chat prompt templates
template = "You're a model that ranks user input according to how they are related to this range of topics and expertise, including crypto, forex trading, options, stocks, defi, forex, and commodities. Rank on a scale of 1 - 10, with 1 being not related at all and 10 being very related to finance. Respond 'This question is not related to finance' when it ranks below 4."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Combine the system and human messages into a chat prompt
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Get a chat completion from the formatted messages
formatted_messages = chat_prompt.format_prompt(text="I love programming.")
formatted_messages.to_messages()

from langchain import LLMChain

# Create an LLMChain filter using the chat prompt and the OpenAI model
llm_chain_filter = LLMChain(prompt=chat_prompt, llm=llm)

# Function to ask a question and get the answer using the LLMChain filter
def answer_question(question, chain=llm_chain_filter):
    print(chain.run(question))

# Example question
question = "What is a lot size?"
answer_question(question)

# The Second Preset

# Define the chat prompt templates for the second preset
template_preset = "You're a finance chatbot that answers people's questions relating only to finance. Your range of topics and expertise includes crypto, stocks, defi, forex, and commodities."
system_message_prompt_preset = SystemMessagePromptTemplate.from_template(template_preset)
human_template_preset = "{text}"
human_message_prompt_preset = HumanMessagePromptTemplate.from_template(human_template_preset)

# Combine the system and human messages into a chat prompt for the second preset
chat_prompt_preset = ChatPromptTemplate.from_messages([system_message_prompt_preset, human_message_prompt_preset])

# Get a chat completion from the formatted messages for the second preset
formatted_messages_preset = chat_prompt_preset.format_prompt(text="I love programming.")
formatted_messages_preset.to_messages()

# Create another LLMChain using the chat prompt for the second preset and the OpenAI model
llm_chain_preset = LLMChain(prompt=chat_prompt_preset, llm=llm)

# Function to ask a question and get the answer using the LLMChain for the second preset
def answer_question_preset(question, chain=llm_chain_preset):
    print(chain.run(question))

# Example question for the second preset
question_preset = "What is today's date?"
answer_question_preset(question_preset)

# Example question for the second preset using the defined function
answer_question_preset("What is a lot?")


