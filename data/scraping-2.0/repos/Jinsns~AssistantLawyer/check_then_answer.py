'''
总流程：
    用户：输入问题
    向量匹配相关的法条，
    问llm：根据这些法条[]，回复，要求客观真实有理有据，不允许编造事实，人名等。
    LLM：输出最终答案

'''


from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import openai
from tenacity import (retry, stop_after_attempt, wait_random_exponential)
import config



# create config.py or write your openai api_key next line
# openai.api_key = config.openai_api_key
openai_api_key = ""


def query_vector_store(query, vector_store_path, embeddings):
    vector_store = FAISS.load_local(folder_path=vector_store_path, embeddings=embeddings)
    search_result = vector_store.similarity_search_with_score(query=query, k=8)
    retrieved_laws = []
    for (doc, l2) in search_result:
        retrieved_law = doc.page_content
        if len(retrieved_law) > 500:
            retrieved_law = retrieved_law[:500]
        retrieved_laws.append(retrieved_law)
        print(doc.page_content, ":", l2)

    return retrieved_laws
    return "\n-----------------------------\n\n".join(retrieved_laws)



@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    """
        in case of network instability, retry
    """
    return openai.ChatCompletion.create(**kwargs)

def ask_gpt(input):
    response = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},

            {"role": "user",
             "content": input}
        ],
        # top_p=0.1,
    )
    completion = response['choices'][0]["message"]["content"]
    # useful_vector_stores_list = eval(completion)
    return completion

if __name__ == "__main__":
    # query = "施工方超过国家规定标准排放噪声，是否应当承担责任?"
    query = "谁可以申请撤销监护人的监护资格?"
    embedding_model_name = "models/embedding_models/text2vec"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    retrieved_laws = query_vector_store(query=query, vector_store_path="data/vector_stores/laws_vector_store", embeddings=embeddings)
    input = (
        f"问题：{query} \n "
        f"为了回答这个问题，我们检索到相关法条如下：\n"
        f"{''.join(retrieved_laws)}\n"
        f"利用以上检索到的法条，请回答问题：{query}\n"
        f"要求逻辑完善，有理有据，不允许伪造事实。"
    )
    completion = ask_gpt(input=input)
    print("答案：", completion)


