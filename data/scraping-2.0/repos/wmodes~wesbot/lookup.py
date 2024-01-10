"""
lookup.py - A Python module for a Lookup class handling information retrieval.

This module defines a Lookup class that constructs a lookup data structure and provides functions for retrieving information about entities.

Features:
- Initializes with a dictionary containing lookup data.
- Offers a method to look up information given a function and args.
- Returns information or an error message if the entity is not found.

Author: Wes Modes
Date: 2023
"""

import openai
import logging
import config
import sys
import json
import pandas as pd
import ast  # for converting embeddings saved as strings back to arrays
from scipy import spatial  # for calculating vector similarities for search
from helpers import super_strip
sys.path.append('lookup')
import lookup_index
import lookup_contents  

# Create a logger instance for the 'lookup' module
logger = logging.getLogger('lookup_logger')

class Lookup:

    def __init__(self):
        self.index = lookup_index.lookup_index
        self.data = lookup_contents.lookup_contents
        self.df = pd.read_csv(config.LOOKUP_EMBED_FILE)
        self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)

    def strings_ranked_by_relatedness(self, query: str, top_n: int = config.LOOKUP_MAX_NUM) -> tuple[list[str], list[float]]:
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y)
        query_embedding_response = openai.Embedding.create(model=config.LOOKUP_EMBED_MODEL, input=query)
        query_embedding = query_embedding_response["data"][0]["embedding"]
        strings_and_relatednesses = [(row["text"], relatedness_fn(query_embedding, row["embedding"])) for i, row in self.df.iterrows()]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]

    def num_tokens(self, text: str, model: str = config.LOOKUP_EMBED_MODEL) -> int:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    def lookup_data(self, prompt: str) -> list[str]:
        """        
        Use embeddings to find relevant lookup records
        
        We return the top config.NUM_RESULTS 
        or the results with a relatedness score of at least config.MIN_RELATEDNESS
        whichever is least
        as an array of strings

        """
        data, relatednesses = self.strings_ranked_by_relatedness(prompt, top_n=config.LOOKUP_MAX_NUM)
        # trim strings that are under config.MIN_RELATEDNESS
        for i, relatedness in enumerate(relatednesses):
            # print(f"\nrelatedness: {relatedness}\ndata: {data[i]}")
            if relatedness < config.LOOKUP_MIN_RELATEDNESS:
                data = data[:i]
                relatednesses = relatednesses[:i]
                break
        return data

    def lookup(self, prompt):
        """
        Look up information given a function and args.

        Args:
          function: The name of the function to call.
          args: A list of arguments to pass to the function.

        Returns:
          The result of the function call.
        """
        # print("lookup() what did we get? ", function_name, args)
        # if the function is lookup_person
        # convert args to a dictionary
        # args_dict = json.loads(args)
        # context = args_dict.get('context').lower()

        # Use embeddings to find the relevant lookup records
        #   We return the top config.NUM_RESULTS 
        #   or the results with a relatedness score of at least config.MIN_RELATEDNESS
        #   whichever is least
        data = self.lookup_data(prompt)

        # if the data array is empty
        if not data:
            # we return nothing
            return None
        # if we have data
        else:
            new_prompt = config.LOOKUP_CAVEAT
            for string in data:
                next_datum = f'\n\Relevant data:\n"""\n{string}\n"""'
                # we should probably check to make sure we are not over the token limit, 
                # but we can do it in chatbot.py
                new_prompt += next_datum
            new_prompt += f"\n\nUser prompt: {prompt}"
            # print(f"\nAfter lookup, here's the new prompt: {new_prompt}")
            return new_prompt
        