import os
from openai import OpenAI
from langchain.document_loaders import UnstructuredMarkdownLoader
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import numpy as np
from ast import literal_eval

from preprocess import (
    process_markdown_batch,
    iterate_folder_files,
    process_files_batch,
)


_ = load_dotenv(find_dotenv())  # read local .env file
client = OpenAI()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")


# model = "gpt-3.5-turbo-1106" or "gpt-4-1106-preview"
def get_completion(prompt, model="gpt-3.5-turbo-1106"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    return response.choices[0].message.content


# Function to create embeddings
def create_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def cosine_similarity(vec1, vec2):
    cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim


def find_most_k_similar(embedding, embeddings_dict, k=5):
    # 初始化一个空的列表来存储相似度和文件名
    similarities = []
    query_embedding = np.array(embedding)

    for filename, file_embedding in embeddings_dict.items():
        # 将文件嵌入向量转换为适合计算的格式
        file_embedding_reshaped = np.array(file_embedding)
        # 计算余弦相似度
        similarity = cosine_similarity(query_embedding, file_embedding_reshaped)
        # 将相似度和文件名添加到列表中
        similarities.append((similarity, filename))

    # 根据相似度排序，取前k个最相似的文件
    similarities.sort(reverse=True)
    most_k_similar = similarities[:k]

    return most_k_similar


def main():
    all_markdown_files = iterate_folder_files(
        root_directory="MarkdownFiles", markdown_files_to_process=[]
    )
    print("all_markdown_files: ", all_markdown_files)

    docs_files = process_files_batch(
        process_markdown_batch,
        all_markdown_files,
        batch_size=1,
        docs=[],
        processed_files=0,
    )

    docs_embeddings = {}
    for doc in docs_files:
        doc_filename = doc.metadata["source"].split("/")[-1].split(".")[0]
        docs_embeddings[doc_filename] = create_embedding(doc.page_content)
    # print(f"docs_embeddings is: {docs_embeddings}")

    query = (
        "I have a A100 GPU and 32GB memory, I want to train a object detection model."
    )

    k = 3
    query_embedding = create_embedding(text=query)
    most_k_similar = find_most_k_similar(query_embedding, docs_embeddings, k)
    print(f"Most {k} likely model are: {most_k_similar}")
    models = [model[1] for model in most_k_similar]

    # prompt：结合models，构建prompt
    model_select_prompt = f"""
    Your task is to identify the most suitable model for the following query, considering a wide range of available models and then compare your choice with the models listed in `{models}`. The query is enclosed in triple backticks.

    Consider these aspects when selecting and comparing the models:
    1. Computational Resources: Assess the required processors (e.g., GPU, TPU, CPU) and memory size.
    2. User Requirements: Take into account the user's needs, focusing on accuracy, latency, and other relevant factors.
    3. Data Type and Scale: Evaluate the compatibility of the model with the type of data (text, image, audio, video) and its scale and complexity.
    4. Task Type: Determine the model's suitability for specific tasks like translation, speech recognition, code generation, etc.
    5. Model Type: Consider the architecture of the model (transformer, CNN, RNN, etc.) and its relevance to the requirements.

    After selecting the most appropriate model based on your knowledge, compare it with the models in `{models}`.
    
    DO not give me explanation information. Only Output a list of the models after you selected and compared. eg. ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview']. If there are no models suitable, output an empty list [].
    
    Query text: ```{query}```
    """
    #  After selecting the most appropriate model based on your knowledge, compare it with the models in `{models}`. If the model you selected is not in `{models}`, explain why it is more suitable than the ones listed.
    #  Output your findings in JSON format with the following keys: chosen_model, compared_models, query, reason for choice.

    model_select_response = get_completion(model_select_prompt)
    print("model_select_response:\n ", model_select_response)
    model_selected_list = literal_eval(model_select_response)
    print("model_selected_list: ", model_selected_list)

if __name__ == "__main__":
    main()
