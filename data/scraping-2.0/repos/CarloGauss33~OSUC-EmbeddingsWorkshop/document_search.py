import openai
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embeddings_vector_for_texts(texts):
    response = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=texts
    )

    return response.data

def create_chat_response(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0.2,
        messages=messages
    )

    return response.choices[0].message.content

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


texts = []
texts_path = "texts/"
for filename in os.listdir(texts_path):
    with open(os.path.join(texts_path, filename), 'r') as f:
        texts.append(f.read())


embeddings_file = "embeddings.npy"
if os.path.isfile(embeddings_file):
    embeddings = np.load(embeddings_file, allow_pickle=True)
else:
    embeddings = []

    for i in range(0, len(texts), 3):
        embeddings.extend(get_embeddings_vector_for_texts(texts[i:i+3]))
    np.save(embeddings_file, embeddings)


most_similar_corpus = get_most_similar_text("Como escapar de la dictadurav", embeddings, texts)
context = {'role': 'system', 'content': 'Resume el siguiente texto'}
messages = [context, {'role': 'user', 'content': most_similar_corpus}]

print(create_chat_response(messages))


