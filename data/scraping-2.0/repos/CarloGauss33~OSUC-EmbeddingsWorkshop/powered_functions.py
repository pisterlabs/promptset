import openai
import os
import numpy as np
from dotenv import load_dotenv
from services.buda_api import get_bitcoin_price, buy_bitcoin
from services.mailer_manager import send_email
from services.weather import get_weather
from services.jokes import get_joke

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def cosine_similarity(data1, data2):
    vector1 = data1.embedding
    vector2 = data2.embedding

    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def get_most_similar_text(text, embeddings, texts):
    text_vector = get_embeddings_vector_for_texts([text])[0]

    similarities = []
    for i, embedding in enumerate(embeddings):
        similarities.append(cosine_similarity(text_vector, embedding))

    return texts[np.argmax(similarities)]


def get_embeddings_vector_for_texts(texts):
    response = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=texts
    )

    return response.data

def create_chat_function_response(messages, model="gpt-4-0613"):
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0.1,
        messages=messages
    )

    return response.choices[0]

codes = []
codes_path = "services/"
for filename in os.listdir(codes_path):
    if filename[-3:] != ".py":
        continue
    with open(os.path.join(codes_path, filename), 'r') as f:
        codes.append(f.read())


embeddings_file = "code_embeddings.npy"
if os.path.isfile(embeddings_file):
    embeddings = np.load(embeddings_file, allow_pickle=True)
else:
    embeddings = []

    for i in range(0, len(codes), 3):
        embeddings.extend(get_embeddings_vector_for_texts(codes[i:i+3]))
    np.save(embeddings_file, embeddings)


def find_and_call_function_through_gpt(message, embeddings, texts):
    message_embedding = get_embeddings_vector_for_texts([message])[0]
    most_similar_text = get_most_similar_text(message, embeddings, texts)

    messages = [
        { 'role': 'system', 'content': 'Transform all the functions of the code to a GPT3-5 Functions capable. For example: {"name": "FUNCTION_NAM", "description": "DESCRIPTION", "parameters": { "type": "TYPE", "properties": { PROPERTIES } }' },
        { 'role': 'user', 'content': most_similar_text } ]

    formatted_code = create_chat_function_response(messages).message.content

    new_messages = [
        { 'role': 'system', 'content': 'Generate a function call according to the user input. The next system message is the code. answer just with the function call' },
        { 'role': 'user', 'content': formatted_code },
        { 'role': 'user', 'content': message } ]

    response = create_chat_function_response(new_messages)
    function_call = response.message.content
    response_f = exec(function_call)



user_input = input("Ingrese su pregunta: ")
find_and_call_function_through_gpt(user_input, embeddings, codes)
