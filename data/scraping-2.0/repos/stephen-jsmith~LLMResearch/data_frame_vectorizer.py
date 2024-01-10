import markdown2
from bs4 import BeautifulSoup
from transformers import GPT2TokenizerFast
import numpy as np
import openai
from openai import OpenAI
import os
import pandas as pd
import pickle
import numpy as np
from nltk.tokenize import sent_tokenize
import glob

EMBEDDING_MODEL = "text-embedding-ada-002"

# Authenticate with OpenAI API
with open("apiKeys.txt", "r") as temp:
    apiKey = temp.read()
client = OpenAI(api_key=apiKey)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

MAX_WORDS = 500


def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))


def get_embedding(text: str, model: str = EMBEDDING_MODEL):
    result = client.embeddings.create(model=model, input=text).data[0].embedding
    return result


def compute_doc_embeddings(df: pd.DataFrame):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {idx: get_embedding(r.content) for idx, r in df.iterrows()}


def vectorize_data(inputDir: str, outputDir: str, ignoreDuplicates: bool = True) -> list:
    """
    Takes in a directory of markdown files to vectorize

    ##### Args #####
    :type dir: str
    :arg dir: Directory of the files you wish to vectorize
    """

    ret_list = []
    filenames = []

    for context in os.listdir(inputDir):
        # Open the markdown file
        with open(os.path.join(inputDir, context), "r") as file:
            content = file.read()

        # Use markdown2 to convert the markdown file to html
        html = markdown2.markdown(content)

        # Use BeautifulSoup to parse the html
        soup = BeautifulSoup(html, "html.parser")

        # Initialize variables to store heading, subheading, and corresponding paragraphs
        headings = []
        paragraphs = []

        data = []

        # Iterate through the tags in the soup
        for tag in soup.descendants:
            # Check if the tag is a heading
            if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                # When the next heading is encountered, print the heading, subheading, and corresponding paragraphs
                if headings and paragraphs:
                    hdgs = " ".join(headings)
                    para = " ".join(paragraphs)
                    data.append([hdgs, para, count_tokens(para)])
                    headings = []
                    paragraphs = []
                # Add to heading
                headings.append(tag.text)
            # Check if the tag is a paragraph
            elif tag.name == "p":
                paragraphs.append(tag.text)

        df = pd.DataFrame(data, columns=["heading", "content", "tokens"])
        df = df[df.tokens > 40]
        df = df.reset_index().drop("index", axis=1)  # reset index
        df.head()

        vector_embedding = compute_doc_embeddings(df)

        df["vector_embedding"] = pd.Series(vector_embedding)
        
        ret_list.append(df)
        filenames.append(context)
        
        '''df.to_csv(os.path.join(outputDir, context.rsplit(".", 1)[0] + ".csv"))'''
        print(f"Successfully vectorized {context}!")

    return ret_list, filenames


vectorize_data("md_files", "vectorized_dataframes")
