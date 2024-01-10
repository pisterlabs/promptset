# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""This module if for chain summarization use case"""

import warnings
warnings.filterwarnings("ignore")

import os

from langchain.chains.summarize import load_summarize_chain

import get_data
import models
import text_preparation

# Get the absolute path by joining the current directory with the relative path
current_directory = os.path.dirname(os.path.abspath(__file__))
# The relative path from the current location to the target file
txt_relative_path = "../data_files/alice_in_wonderland.txt"
txt_file_path = os.path.join(current_directory, txt_relative_path)


if __name__ == "__main__":
    docs = get_data.load_local_file(txt_file_path)
    chunks = text_preparation.split_docs_recursively(docs=docs, chunk_size=500)[:2]  # Save tokens ...
    llm = models.set_openai_chat_model(max_tokens=500)
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    summary = chain.run(chunks)
    print("Summary: \n", summary)
