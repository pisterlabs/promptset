"""A core module to process website text data.
"""

import os
import pandas as pd
from urllib.parse import urlparse
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings
import numpy as np
from src.utility.utils import config
from src.utility import constants
from src.utility.nlp_text_cleaner import remove_newlines
from src.data_collection import web_crawler


openai.api_key = config.get(constants.OPENAI_API_KEY)

class DataProcessor:
    """A class having data processing methods
    """

    def __init__(self,full_url) -> None:
        """A Dataprocessor constructor

        Args:
            full_url (str): Full website URL
        """   
        # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 500
        #self.domain = config.get(constants.DOMAIN)
        self.full_url = full_url
        self.local_domain = urlparse(self.full_url).netloc
        

    def create_dataset_from_text_files(self):
        """A method to create data file from website text files.
        """

        # Create a list to store the text files
        texts = []

        # Get all the text files in the text directory
        for file in os.listdir(config.get(constants.TEXT_DATA_PATH) + self.local_domain + "/"):
            # Open the file and read the text
            with open(
                config.get(constants.TEXT_DATA_PATH) + self.local_domain + "/" + file,
                "r",
                encoding="UTF-8",
            ) as f:
                text = f.read()

                # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
                texts.append(
                    (
                        file[11:-4]
                        .replace("-", " ")
                        .replace("_", " ")
                        .replace("#update", ""),
                        text,
                    )
                )

        # Create a dataframe from the list of texts
        df = pd.DataFrame(texts, columns=["fname", "text"])

        # Set the text column to be the raw text with the newlines removed
        df["text"] = df.fname + ". " + remove_newlines(df.text)
        df.to_csv(config.get(constants.EMBEDDINGS_DATA_PATH) + self.local_domain + "/scraped.csv")
 

    def tokenize_texts(self):
        """A mthod to tokenize text data

        Returns:
            dataframe (pandas): A dataframe with tokenized text
        """        
        df = pd.read_csv(config.get(constants.EMBEDDINGS_DATA_PATH) + self.local_domain + "/scraped.csv", index_col=0)
        df.columns = ["title", "text"]

        # Tokenize the text and save the number of tokens to a new column
        df["n_tokens"] = df.text.apply(lambda x: len(self.tokenizer.encode(x)))

        # Visualize the distribution of the number of tokens per row using a histogram
        # df.n_tokens.hist()
        return df


    # Function to split the text into chunks of a maximum number of tokens
    def split_into_many(self,text):
        """A method to split the text into predefined samll chunks

        Args:
            text (str): A text data

        Returns:
            chunks (list): A list of chunks of split text data
        """        
        # Split the text into sentences
        sentences = text.split(". ")

        # Get the number of tokens for each sentence
        n_tokens = [len(self.tokenizer.encode(" " + sentence)) for sentence in sentences]

        chunks = []
        tokens_so_far = 0
        chunk = []

        # Loop through the sentences and tokens joined together in a tuple
        for sentence, token in zip(sentences, n_tokens):
            # If the number of tokens so far plus the number of tokens in the current sentence is greater
            # than the max number of tokens, then add the chunk to the list of chunks and reset
            # the chunk and tokens so far
            if tokens_so_far + token > self.max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

            # If the number of tokens in the current sentence is greater than the max number of
            # tokens, go to the next sentence
            if token > self.max_tokens:
                continue

            # Otherwise, add the sentence to the chunk and add the number of tokens to the total
            chunk.append(sentence)
            tokens_so_far += token + 1

        return chunks


    def create_initial_dataset(self,df):
        """A method to create initial dataframe from text

        Args:
            df (pandas): A dataframe with text data

        Returns:
            df (pandas): A dataframe with tokenized and chunk text data
        """        
        shortened = []

        # Loop through the dataframe
        for row in df.iterrows():
            # If the text is None, go to the next row
            if row[1]["text"] is None:
                continue

            # If the number of tokens is greater than the max number of tokens, split the text into chunks
            if row[1]["n_tokens"] > self.max_tokens:
                shortened += self.split_into_many(row[1]["text"])

            # Otherwise, add the text to the list of shortened texts
            else:
                shortened.append(row[1]["text"])

        ################################################################################
        ### Step 9
        ################################################################################

        df = pd.DataFrame(shortened, columns=["text"])
        df["n_tokens"] = df.text.apply(lambda x: len(self.tokenizer.encode(x)))
        # df.n_tokens.hist()
        return df


    def create_embeddings(self,df):
        """A method to create text embeddings of text chunks

        Args:
            df (pandas dataframe): A dataframe with tokenized and chunk text data
        """        
        # Note that you may run into rate limit issues depending on how many files you try to embed
        # Please check out our rate limit guide to learn more on how to handle
        # this: https://platform.openai.com/docs/guides/rate-limits

        df["embeddings"] = df.text.apply(
            lambda x: openai.Embedding.create(input=x, engine="text-embedding-ada-002")[
                "data"
            ][0]["embedding"]
        )
        df.to_csv(config.get(constants.EMBEDDINGS_DATA_PATH) + self.local_domain + "/embeddings.csv")
        # df.head()


    def get_embeddings(self):
        """A method to retun embeddings dataframe

        Returns:
            df (pandas dataframe): A dataframe with text embeddings
        """
        if not os.path.exists(
            config.get(constants.EMBEDDINGS_DATA_PATH)
            + self.local_domain
            + "/"
        ):
            web_crawler.crawl(self.full_url)

            self.create_dataset_from_text_files()
            dataset = self.tokenize_texts()
            dataset = self.create_initial_dataset(dataset)
            self.create_embeddings(dataset)

        df = pd.read_csv(config.get(constants.EMBEDDINGS_DATA_PATH) + self.local_domain + "/embeddings.csv", index_col=0)
        df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)
        
        return df
 