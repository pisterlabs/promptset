# imports
import ast  # for converting embeddings saved as strings back to arrays
import os
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.organization = os.getenv('OPENAI_ORGANIZATION')

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"     # can handle 北邮/港大/南开+宿舍
TOKEN_BUDGET = 4096
# GPT_MODEL = "gpt-3.5-turbo-16k"   # can't handle 北邮/港大/南开+宿舍
# TOKEN_BUDGET = 16000

all_df = []
for csv in os.listdir("embedding"):
    all_df.append(pd.read_csv("embedding/" + csv))
df = pd.concat(all_df)
df['embedding'] = df['embedding'].apply(ast.literal_eval)   # convert embeddings from CSV str type back to list type

links_df = pd.read_csv("csv_other/links.csv")


def strings_ranked_by_relatedness(
        query,
        relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n = 30
    ):
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [(row["text"], relatedness_fn(query_embedding, row["embedding"])) for i, row in df.iterrows()]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text):
    encoding = tiktoken.encoding_for_model(GPT_MODEL)
    return len(encoding.encode(text))


def query_message(query, token_budget, query_history = ""):
    introduction = "根据以下提供的“大学信息”作为资料进行回答。如果你不知道答案，就说“我不知道”，不要擅自进行回答。问题在最后。"
    message = introduction
    question = f"\n\n问题: {query}"
    paragraph, relatednesses = strings_ranked_by_relatedness(query_history + query)
    for para in paragraph:
        next_article = f'\n\n大学信息:\n"""\n{para}\n"""'
        if num_tokens(message + next_article + question) > token_budget:
            print("MAXOUT!!!!!!!!!!!!!!!!!!!!!!")
            break
        else:
            message += next_article
    return message + question


def ask(query, history = []):
    token_budget = TOKEN_BUDGET - 500
    messages = [
        {"role": "system", "content": "回答关于大学校园或专业的问题。"},
    ]
    query_history = ""
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
        token_budget -= num_tokens(q + a)
        query_history += q
    
    message = query_message(query, token_budget, query_history)
    messages.append({"role": "user", "content": message + "\n\nMUST ANSWER IN ENGLISH\nMUST ANSWER IN ENGLISH\nMUST ANSWER IN ENGLISH\nMUST ANSWER IN ENGLISH"})

    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


def add_link(paragraph):
    all_links = []
    link_unique = set()
    for i, row in links_df.iterrows():
        uni = row["university"]
        link = row["link"]
        if uni in paragraph and not link in link_unique:
            all_links.append((uni, link))
            link_unique.add(link)

    if 0 < len(all_links) and len(all_links) < 5:
        # paragraph += "\n\n相关文档："
        paragraph += "\n\nReference："
        for uni, link in all_links:
            paragraph += f"\n{uni}: {link}"
    return paragraph


if __name__ == "__main__":
    print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}    GPT_MODEL: {GPT_MODEL}    TOKEN_BUDGET: {TOKEN_BUDGET}")
    # q = "北理工学习氛围怎么样"
    # res = ask(q)
    # res = add_link(res)
    # print(res)
    history = []
    while True:
        # if len(history) > 3:
        #     history = history[-3:]
        q = input("--------------------------------------------\n问题：")
        # print(f"history: {len(history)} {history}")
        a = ask(q, history)
        print(add_link(a))
        # history.append((q, a))
