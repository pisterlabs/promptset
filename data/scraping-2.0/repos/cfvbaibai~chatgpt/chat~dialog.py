import json

import numpy as np
import openai
import pandas as pd
from numpy import ndarray
from openai import OpenAI

CHAT_COMPLETION_MODEL = "gpt-3.5-turbo"
COMPLETIONS_MODEL = 'text-davinci-003'
EMBEDDING_MODEL = "text-embedding-ada-002"

client = OpenAI()

df_qa = pd.read_csv('data/md_QA_embedded.csv')
df_qa['embedding'] = df_qa['embedding'].apply(lambda x: json.loads(x))


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = client.embeddings.create(
        model=model,
        input=text
    )
    return result.data[0].embedding


def vector_similarity(x: list[float], y: list[float]) -> ndarray:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def calc_embeddings(df):
    df['embedding'] = df['Questions'].apply(lambda s: get_embedding(s))
    df.to_csv('../data/embedded.csv', index=False)
    return df.head()


class Dialog:
    @staticmethod
    def _get_most_similar_qa(question):
        q_embedding = get_embedding(question)
        df = df_qa.copy()
        df['similarity'] = df['embedding'].apply(lambda x: vector_similarity(x, q_embedding))
        sorted_df = df.sort_values(by='similarity', ascending=False)
        best_q, best_a, similarity = sorted_df[['Questions', 'Answers', 'similarity']].iloc[0]
        return best_q, best_a, similarity

    @staticmethod
    def _get_prompt_by_best_qa(q, best_q, best_a, similarity):
        lines = []
        if similarity > 0.9:
            lines.append(f"Q: {best_q}")
            lines.append(f"A: {best_a}")
            lines.append("")
        lines.append(f"Q: {q}")
        lines.append("A: ")
        prompt = "\n".join(lines)
        print(prompt)
        return prompt

    def __init__(self):
        self.messages = []
        self.token_counts = []

    def _get_total_token_counts(self):
        return sum(self.token_counts)

    def set_cpu_role(self, message):
        self.messages = [item for item in self.messages if item["role"] != "system"]
        self.messages.insert(0, {"role": "system", "content": message})
        self._print_state("set_cpu_role")

    def _print_state(self, title):
        print("========", title, "========")
        print("total token count: ", self._get_total_token_counts())
        print("messages count: ", len(self.messages))
        print("token_counts: ", str(self.token_counts))
        print("===================================")

    def ask(self, question):
        best_q, best_a, similarity = Dialog._get_most_similar_qa(question)
        total_tokens = 0
        if similarity > 0.925:
            self.messages.append({"role": "user", "content": question})
            self.token_counts.append(0)
            answer = best_a
        else:
            prompt = Dialog._get_prompt_by_best_qa(question, best_q, best_a, similarity)
            self.messages.append({"role": "user", "content": prompt})
            result = client.chat.completions.create(
                model=CHAT_COMPLETION_MODEL,
                messages=self.messages
            )
            total_tokens = result.usage.total_tokens
            self.token_counts.append(total_tokens - self._get_total_token_counts())
            print(result)
            answer = result.choices[0].message.content
        self.messages.append({"role": "assistant", "content": answer})
        self._print_state("THIS ROUND")

        overflow = total_tokens >= 4097
        if overflow:
            while self._get_total_token_counts() >= 2048:
                self.messages.pop(0)
                self.messages.pop(0)
                self.token_counts.pop(0)
                self._print_state("AFTER CUT")

        return {"final_a": answer, "best_q": best_q, "best_a": best_a, "similarity": similarity, "overflow": overflow}
