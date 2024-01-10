from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, \
    HumanMessagePromptTemplate
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import json
import numpy as np
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import openai
import tiktoken

openai.api_key = "yourAPI_KEY"
chat = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key="yourAPI_KEY")

messages = [
    SystemMessage(content="You are a manager of my google account data"),
]

with open("data.json", encoding='utf-8') as jsonfile:
    data = json.load(jsonfile)

def get_context(inputPrompt, embeddings):
    search_term_vector = embeddings
    for item in data:
        item['embeddings'] = np.array(item['embeddings'])

    for item in data:
        item['similarities'] = cosine_similarity(item['embeddings'], search_term_vector)

    sorted_data = sorted(data, key=lambda x: x['similarities'], reverse=True)
    context = ''
    referencs = []
    # for i in sorted_data[0]:
    context += sorted_data[0]['chunk_text'] + '\n'

    return context



total_token = len(messages[0].content.split(" "))

while True:
    user_input = input("You: ")

    embd = get_embedding(user_input, engine='text-embedding-ada-002')

    context = get_context(user_input, embd)

    messages.append(HumanMessage(
        content="context: \n" + context + "\n\nplease answer the following question using the above given context\n\n" + user_input))
    messages[-1].content = messages[-1].content.replace("\n", " ")
    n_token = len(messages[-1].content.split(" ")) * 100 / 75
    total_token += n_token
    while total_token > 3500:
        n_token = len(messages[1].content.split(" ")) * 100 / 75
        total_token = total_token - n_token
        messages.pop(1)
    ai_response = chat(messages=messages).content
    print("Chatbot: ", ai_response)
    messages.append(AIMessage(content=ai_response))
    n_token = len(messages[-1].content.split(" ")) * 100 / 75
    total_token += n_token

