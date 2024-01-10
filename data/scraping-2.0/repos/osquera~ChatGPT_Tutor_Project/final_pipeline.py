import tensorflow as tf
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
# for not seeing a warning message
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
import openai
import os
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_text_embedding(text: str):
    # Load pre-trained model and tokenizer
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
    embedding = model.encode(text, convert_to_tensor=True)

    return embedding

def semantic_search_model(embedding: np.ndarray, n_contexts: int = 5) -> str:
    """
    This function takes in a query embedding and finds the most relevant document by using the ANN
    :param n_contexts:  The number of contexts to return
    :param embedding: The query embedding of the query of dimension 768
    :return: The most relevant n contexts
    """
    # load the dataset
    df = pd.read_pickle('./Data_Generation/df_pickle/final_02450_emb.pkl')
    # make the df only contain the unique contexts
    df = df.drop_duplicates(subset=['context'])
    context_embeddings = np.stack(df['context_embedding'].to_numpy())

    # Compute cosine similarity
    x = np.array(embedding)
    y = context_embeddings
    cos_sim = np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y, axis=1))
    # Get the n most relevant contexts
    index = np.argsort(cos_sim, axis=0)[-n_contexts:]

    best_ctx_lst = [df.iloc[i]["context"] for i in index]
    best_ctx = '\n'.join(best_ctx_lst)

    return best_ctx


def answer_generation(query: str, context: str = "", chatgpt_prompt = "You are a Teachers Assistant and you should answer the QUESTION using the information given in the CONTEXT, if the CONTEXT is unrelated, you should ignore it."):

    """
    This function takes in a query and a context and uses the OpenAI API to generate an answer
    :param query:
    :param context:
    :return:
    """
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages = [
            {"role": "user", "content": f'  {chatgpt_prompt},CONTEXT: ```{context}``` QUESTION: ```{query}``` ANSWER:'},
        ]
        )
    except Exception as e:
        return "OPENAI_ERROR:", str(e)

    return completion.choices[0].message.content


def pipeline(query: str, n_contexts: int = 5, chatgpt_prompt = "As a teacher's assistant for a machine learning course, your role is to assist students by answering their questions. You have over 20 years of experience and are regarded as one of the best in the field. You will be presented with a question and given several context paragraphs from the lecture notes. Your task is to answer the question as comprehensively and accurately as possible  Please thoroughly read the context paragraph and determine if any information is relevant to answer the question. Your answer should be based on your own high level of expertise and only use the information in the context paragraph if any is relevant. In case the question does not have any direct reference in the lecture notes, please state that the specific information is not mentioned in the lecture notes and proceed to answer the question based on your knowledge. You may give a long answer if the question requires it."):
    """
    This function is the pipeline for the entire project. It takes in a query and finds the most relevant document.
    and gives it to the OpenAI API to generate a answer
    :param n_contexts: The number of contexts to return
    :param query: The query to search for
    :return:
    """
    # 1. Preprocess the query
    embedding = get_text_embedding(query)
    # 2. Semantic Search
    best_ctx = semantic_search_model(embedding, n_contexts)
    # 3. Answer Generation
    answer = answer_generation(query, best_ctx, chatgpt_prompt=chatgpt_prompt)
    # 4. Return the answer
    return answer, best_ctx



if __name__ == "__main__":
    # Read in the data
    query = "Who is Morten Mørup and Tue Herlau?"
    answer_pipeline = pipeline(query, n_contexts=5)[0]
    print(answer_pipeline)
