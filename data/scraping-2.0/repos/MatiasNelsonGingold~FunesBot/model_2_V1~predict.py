from model import *
from primer_input import *
from params import *


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from params import *


import openai
import os
import sys


def predict_final(query,pipe,chat,prompt_template):

    prediction = pipe.run(
        query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 5}}
    )

    # crear lista del contexto retornado por el retrieval utilizado
    list_of_contextual_ans_retrieval=[]

    for i in range (5):
        list_of_contextual_ans_retrieval.append(prediction['documents'][i].content)


    # #establecer el formato de llamar el openai chat
    preparation_answer_user = prompt_template.format_messages(
                        query=query,
                        list_of_contextual_ans_retrieval=list_of_contextual_ans_retrieval)


    # # Call the LLM to answer the question with the context cited
    answer_user_final = chat(preparation_answer_user)


    return dict(Respuesta = answer_user_final)
