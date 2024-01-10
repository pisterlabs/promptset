import openai
import numpy as np
import pandas as pd
from openai.embeddings_utils import distances_from_embeddings


class ChatPDF:

    def __init__(self, embedded_df: pd.DataFrame) -> None:
        """This model take a DataFrame as input and could answer question related to it with the help of ChatGPT
        
        :param embedded_df: a DataFrame with columns [text, embeddings, num_tokens]
        """

        embedded_df['embeddings'] = embedded_df['embeddings'].apply(np.array)
        self.df = embedded_df

    def find_context(self, question: str, max_len: int = 1800) -> str:
        """Find most related context to the question

        :param question: the question
        :param max_len: the maximum number of tokens of the related context
        :return: the most related context
        """

        print("Searching related context...")

        # embed the question
        embeded_q = openai.Embedding.create(
            input=question,
            engine='text-embedding-ada-002')['data'][0]['embedding']

        # get the distance from the embeddings
        self.df["dist"] = distances_from_embeddings(
            embeded_q, self.df["embeddings"].values, distance_metric='cosine')

        related_context = list()
        cur_len = 0

        # sort and find the closest context
        for i, row in self.df.sort_values("dist", ascending=True).iterrows():

            # count the context length
            cur_len += row["num_tokens"] + 4

            # If the context is too long, break
            if cur_len > max_len:
                break

            # store the related context
            related_context.append(row["text"])

        print("Finish searching!\n")

        return "\n\n###\n\n".join(related_context) + "\n\n###\n"

    def get_answer(self,
                   question: str,
                   max_len: int = 1800,
                   verbose: bool = False,
                   **kwargs) -> str:
        """Get the answer of the question according to this pdf

        :param question: the question
        :param max_len: the maximum number of tokens of the related context
        :param verbose: if true, print the most related context
        :param kwargs: parameters for openai.ChatCompletion.create, except model and messages
        :return: the answer to the question
        """

        related_context = self.find_context(question, max_len)

        if verbose:
            print("Context:\n" + related_context)
            print()

        try:
            full_question = f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {related_context}\n\n---\n\nQuestion: {question}\nAnswer:"
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages=[{
                                                        "role":
                                                        "user",
                                                        "content":
                                                        full_question
                                                    }],
                                                    **kwargs)
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(e)
            return "Failed to answer"
