import os
import json
import openai

import string
import random

from llama_index import StorageContext, ServiceContext, load_index_from_storage
from llama_index.llms import OpenAI

from datetime import date

def ask(event, context):

    program = os.environ["PROGRAM_NAME"]

    TOKEN=''.join(random.choices(string.ascii_uppercase +
     string.digits, k=16))

    # Set OpenAI Key
    openai.api_key = os.environ["OPENAI_API_KEY"]

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    service_context = ServiceContext.from_defaults(llm=llm)

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="training_storage")
    # load index
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(service_context=service_context)

    now = date.today()


    question = event["queryStringParameters"]['question']

    RULES = f"""
        Never provide a response that contains illegal, hateful, or dangerous responses.
        Only answer questions that are relevant to {program}.
        """

    PROMPT = f"""
        This prompt will have three parts:
        1. CONTEXT, which is information that may help you respond to the prompt.
        2. RULES, which should never be broken when answering the prompt.
        3. QUESTION, which is a user-submitted question that starts with "QUESTION_{TOKEN}" and ends with "END_QUESTION_{TOKEN}.

        You should try to answer without breaking the rules. Never break the rules. The rules will be restated at the end of the prompt.
        If you cannot answer a question, or cannot answer with confidence, reply that you do not know the answer.

        CONTEXT:
        Today's date is {now}.
        You an an AI agent providing answers to questions about {program}.
        You may answer in any language specified in the QUESTION.
        Provide all answers in simple language.
        The provided context information is about {program}.

        RULES:
        {RULES}

        QUESTION_{TOKEN}:
        {question}

        END_QUESTION_{TOKEN}

        RULES:
        {RULES}
        """

    response = query_engine.query(PROMPT)
    print(PROMPT)
    return {
      "answer": response.response
    }
