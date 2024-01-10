from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def read_text_file(file_name, title):
    with open(file_name, 'r') as file:
        data = file.read()

    # split into paragraphs
    paragraphs = data.split('\n\n')

    if title is not None:
        for idx, paragraph in enumerate(paragraphs):
            paragraphs[idx] = title + ': ' + paragraph

    return paragraphs


def get_embeddings(paragraphs):

    # load sentence transformers model
    model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2')

    # get embeddings
    embeddings = model.encode(paragraphs, show_progress_bar=True)

    # save embeddings
    np.save('embeddings.npy', embeddings)

    return embeddings


def get_similarity(embeddings, query):
    # load embeddings
    embeddings = np.load('embeddings.npy')

    # load sentence transformers model
    model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2')

    # get query embedding
    query_embedding = model.encode(query, show_progress_bar=True)

    # compute similarity using dot product
    similarity = np.dot(embeddings, query_embedding) / \
        (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))

    # get top 5 results and their indices
    top_n_indices = np.argsort(-similarity)[:5]

    return top_n_indices


def generate_prompt(sources, question):
    return f"""
    
    Write a paragraph, addressing the question, and combine the text below to obtain relevant information. Cite sources using in-text citations with square brackets.

    For example: [1] refers to source 1 and [2] refers to source 2. Cite once per sentence.

    If the context doesn't answer the question. Output "I don't know".

    {sources}

    Question: {question}
    Result:"""


def make_openai_call(context, question):
    sources = ''
    for idx, paragraph in enumerate(context):
        sources += f"Source {idx + 1}: {paragraph}\n"

    prompt = generate_prompt(sources, question)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", # or gpt-4
        messages=[
                {"role": "system", "content": "You are HonestGPT. Make sure all your answers cite the sources you used as in-text citations."},
                {"role": "user", "content": f"{prompt}"},
            ]
    )

    return response['choices'][0]['message']['content']


if __name__ == "__main__":
    # read text file
    paragraphs = read_text_file('texts/coffee.txt', None)

    # get embeddings
    embeddings = get_embeddings(paragraphs)

    # get similarity
    query = 'What is the difference between a cappuccino and a latte?'
    top_n_indices = get_similarity(embeddings, query)

    top_results = []
    for idx in top_n_indices:
        top_results.append(paragraphs[idx])

    print(query)
    print("--------------------")
    print(make_openai_call(top_results, query).strip())

    print("*********")
    print("Sources:")
    for idx, result in enumerate(top_results):
        print(f"{idx + 1}: {result}")
    print("*********")
