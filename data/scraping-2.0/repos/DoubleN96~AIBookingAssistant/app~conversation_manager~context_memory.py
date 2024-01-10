# import gradio as gr
import logging
import os
import openai

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from app.conversation_manager.prompt import ASSISTANT_PROMPT_1, SYSTEM_PROMPT, USER_PROMPT_1, GUIDELINES_PROMPT

MODEL = 'gpt-3.5-turbo'
openai.api_key = os.environ.get('OPEN_API_KEY', "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")


llm = OpenAI(
    model_name=MODEL,
    temperature=0.3,
    openai_api_key=openai.api_key
)

booking_prompt = PromptTemplate(
    input_variables=["product_description"],
    template="Create comma seperated product keywords to perform a query on a airbnb dataset for this user input: "
             "{product_description}",
)


def get_booking_chain():
    logging.info('Building the chatgpt booking assistant object.')
    return LLMChain(llm=llm, prompt=booking_prompt)


chatter = openai.ChatCompletion(
    model_name=MODEL,
    verbose=False
)


def get_response(message_history: list[dict], user_input: str):
    if not message_history:
        message_history = [
            {"role": "system", "content": "You are the assistant answering users questions."},
            {"role": "user", "content": user_input}
        ]
    return chatter.create(
          model="gpt-3.5-turbo",
          messages=message_history
    )


def openai_chat_completion_ner_response(sentence_to_extract: str):
    ner_input = GUIDELINES_PROMPT.format(sentence_to_extract)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_1},
            {"role": "assistant", "content": ASSISTANT_PROMPT_1},
            {"role": "user", "content": ner_input}
        ]
    )
    logging.warning(
        f'NER response is {response}'
    )

    return response['choices'][0]['message']['content'].strip(" \n")
