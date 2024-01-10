import os
import re
import time
import openai
import tiktoken
import pdfplumber
import pandas as pd


class EmbedPDF:

    def __init__(self, file_path: str) -> None:
        """This is a class for converting PDF to a DataFrame with embeddings of the text

        :param file_path: the path of the PDF
        """

        self.file_path = file_path
        self.file_name = os.path.basename(self.file_path).split(".")[0]
        self.contents = list()
        self.embed_df = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def pdf_to_df(self, **kwargs) -> pd.DataFrame:
        """Convert PDF text to pandas DataFrame

        :param kwargs: parameters for pdfplumber.open
        :return: a DataFrame contains Page No. and Contents
        """

        print("\nReading pdf...")

        with pdfplumber.open(self.file_path, **kwargs) as pdf:
            self.pages = pdf.pages

            for i, page in enumerate(self.pages):
                print(f"Reading Page {i+1}...")
                txt = re.sub(r"\s", " ", page.extract_text())
                self.contents.append([i, txt])

        print("Finish reading!\n")

        self.contents[0][1] = (self.file_name + " " + self.contents[0][1]
                               )  # add file name to Page 0

        return pd.DataFrame(self.contents, columns=("Page No.", "Contents"))

    def break_long_text(self, text: str, **kwargs) -> list:
        """Break the long text into chunks of a maximum number of tokens

        :param text: the long text
        :param kwargs: could contain period_type: str and max_token: int
        :return: a list of chunks
        """

        period_type = kwargs.get(
            "period_type")  # default using . to split text
        max_token = kwargs.get("max_token")

        sentences = text.split(period_type)
        chunks = list()
        count_tokens = 0
        to_merge = list()

        num_tokens = [(s, len(self.tokenizer.encode(s))) for s in sentences]

        for s, n in num_tokens:

            # ignore single sentence if its tokens exceed the maximum
            if n > max_token:
                continue

            # concatenate cumulative sentences when the length near maximum
            if count_tokens + n > max_token:
                chunks.append(
                    ((period_type + " ").join(to_merge) + period_type))
                to_merge.clear()
                count_tokens = 0

            to_merge.append(s)
            count_tokens += n + 1

        # concatenate remaining sentences
        if count_tokens:
            chunks.append(((period_type + " ").join(to_merge) + period_type))

        return chunks

    def format_text(self,
                    df: pd.DataFrame,
                    max_token: int = 500,
                    period_type: str = ".") -> pd.DataFrame:
        """Format the text to make each row's length within max_token

        :param df: a DataFrame contains Page No. and Contents
        :param max_token: the maximum number of token of a model for embedding
        :param period_type: "." for English text and "。" for Chinese text
        :return: a DataFrame with text column with each row not exceeding max_token
        """

        print("Formating...")

        df["num_tokens"] = df["Contents"].apply(
            lambda x: len(self.tokenizer.encode(x)))

        to_token = list()
        for row in df.iterrows():

            # skip no content
            if row[1]["Contents"] is None:
                continue

            # split long text
            if row[1]["num_tokens"] > max_token:
                to_token += self.break_long_text(row[1]["Contents"],
                                                 max_token=max_token,
                                                 period_type=period_type)

            # directly add short text
            else:
                to_token.append(row[1]["Contents"])

        print("Finish formating!\n")

        return pd.DataFrame(to_token, columns=["text"])

    def embed(self,
              embed_df: pd.DataFrame,
              limit_per_min: int = 60) -> pd.DataFrame:
        """Embedding the DataFrame

        :param embed_df: a DataFrame with text column
        :param limit_per_min: the rate limit of the number of requests per minute
        :return: a DataFrame contains columns [text, embeddings, num_tokens]
        """

        print("Generating embeddings...")

        num_rows = len(embed_df)
        embed_df.insert(1, "embeddings", None)
        embed_df.insert(2, "num_tokens", None)

        # generate embeddings
        for i in range(0, num_rows, limit_per_min):
            print(
                f"Generating embeddings from rows {i+1} to {min(i+limit_per_min, num_rows)}..."
            )
            embed_df["embeddings"][i:i + limit_per_min] = embed_df["text"][
                i:i + limit_per_min].apply(lambda x: openai.Embedding.create(
                    input=x, engine="text-embedding-ada-002")["data"][0][
                        "embedding"])
            embed_df["num_tokens"][i:i + limit_per_min] = embed_df["text"][
                i:i +
                limit_per_min].apply(lambda x: len(self.tokenizer.encode(x)))

            # the limit of openai.Embedding.create is 60/min
            if i + 60 < num_rows:
                time.sleep(60)

        self.embed_df = embed_df

        print("Finish generating!\n")

        return embed_df
