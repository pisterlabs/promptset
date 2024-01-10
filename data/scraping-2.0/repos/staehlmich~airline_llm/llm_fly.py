#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import openai
from llama_index import download_loader, GPTSimpleVectorIndex

import config

os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Python programm to answer questions about flight data using LLMs.")
    parser.add_argument("prompt", help="Prompt to pass to LLM")
    parser.add_argument("mode", help="Choose 'tuned' to use fine-tuned LLM or 'context' to add the airline data to the prompt.")
    args = parser.parse_args()

    if args.mode == "tuned":
        # Make the completion request using fine-tuned model.
        completion = openai.Completion.create(model=config.model_name,
                                              prompt=args.prompt,
                                              max_tokens=100,
                                              temperature=0.2)
        print(completion.choices[0]["text"])

    if args.mode == "context":

        #Create data loader
        PandasCSVReader = download_loader("PandasCSVReader")
        loader = PandasCSVReader()
        #Load data to input as context in prompt.
        documents = loader.load_data(
            file=Path("data/airlines_delay_small_fm.csv"))
        doc_index = GPTSimpleVectorIndex.from_documents(documents)
        response = doc_index.query(args.prompt)
        print(response)