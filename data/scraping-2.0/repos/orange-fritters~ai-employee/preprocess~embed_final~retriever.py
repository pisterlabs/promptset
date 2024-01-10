import ast
import numpy as np
import pandas as pd
import openai
import tiktoken
openai.api_key_path = "config.txt"


class Embed:
    def __init__(self):
        self.data = pd.read_csv("preprocess/embed_final/final.csv")
        self.title = self.data['title_kor']
        self.embed = np.array(self.data['title_embed'].apply(ast.literal_eval).tolist())
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def get_score(self,
                  query: str):
        query_eng, price = self.convert_query(query)
        query_embed = self.__get_embed(query_eng)
        query_embed = np.array(query_embed)  # 1536, 1
        sim = np.dot(self.embed, query_embed).reshape(-1)  # 462, 1
        return sim

    def experiment(self,
                   query: str,
                   title: str):
        query_embed = self.__get_embed(query)
        query_embed = np.array(query_embed)  # 1536, 1
        sim = np.dot(self.embed, query_embed).reshape(-1)  # 462, 1
        return self.rank_from_title(sim, title)

    def rank_from_title(self,
                        score: np.array,
                        title: str):
        descending_score = np.argsort(score)[::-1]
        title_idx = self.title[self.title == title].index[0]
        rank = np.where(descending_score == title_idx)[0][0] + 1
        top1 = self.title[descending_score[0]]
        return rank, top1

    def get_top_5(self,
                  score: np.array):
        descending_score = np.argsort(score)[::-1]
        # extract descending score indexed rows
        top5 = self.data.iloc[descending_score[:5]]
        return top5

    def get_embed(self, text: str):
        print("To get embed", text)
        result = openai.Embedding.create(
            engine="text-embedding-ada-002",
            input=text)
        return result["data"][0]["embedding"]

    def convert_query(self, query: str) -> (str, float):
        content = f"""
        Translate the following Korean sentence into English.
        
        Korean:
            ```{query}```
        Output:
            ```
            {{ translation }}
            ```
        """
        tokens = self.__count_token(content)
        response = openai.ChatCompletion.create(
            model=(ver := "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": content}],
        )
        price = 0.0015 * tokens * 1.3
        price += 0.002 * self.__count_token(response['choices'][0]['message']['content']) * 1.3
        return response['choices'][0]['message']['content'], price

    def __get_embed(self, text: str):
        print(text)
        result = openai.Embedding.create(
            engine="text-embedding-ada-002",
            input=text)
        return result["data"][0]["embedding"]

    def __count_token(self, text, input=True):
        return len(self.encoding.encode(text))
