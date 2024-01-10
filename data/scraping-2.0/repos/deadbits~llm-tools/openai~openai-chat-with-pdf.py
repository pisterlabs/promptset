#!/usr/bin/env python3
##
# github.com/deadbits
## 
import os
import sys
import logging
import argparse

import openai

from pathlib import Path

from langchain import OpenAI

from llama_index import GPTSimpleVectorIndex
from llama_index import LLMPredictor
from llama_index import ServiceContext
from llama_index import download_loader


# disable some of the more verbose llama_index logging
logging.basicConfig(level=logging.CRITICAL)

# change this to your preferred model
MODEL_NAME = 'gpt-3.5-turbo'


def load_document(fpath):
    print(f'[status] loading document: {fpath}')
    PDFReader = download_loader('PDFReader')
    loader = PDFReader()
    docs = loader.load_data(file=Path(fpath))
    return docs


def chat(fpath, api_key):
    docs = load_document(fpath)
    llm = LLMPredictor(llm=OpenAI(openai_api_key=api_key, temperature=0.7, model_name=MODEL_NAME))
    ctx = ServiceContext.from_defaults(llm_predictor=llm, chunk_size_limit=1024)
    index = GPTSimpleVectorIndex.from_documents(docs, service_context=ctx)

    print('[status] ready to chat\nhit ctrl+c or type "exit" to quit\n')

    try:
        while True:
            prompt = input("💀 >> ")
            if prompt.lower() == "exit" or prompt.lower() == 'quit':
                break

            response = index.query(prompt)
            response = str(response)
            if response.startswith('\n'):
                response = response[1:]
            print(f'🤖 >> {response}')

    except KeyboardInterrupt:
        print('exiting ...')
        sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create and index embeddings from directory of files')

    parser.add_argument(
        '-f', '--file',
        help='PDF document to chat with',
        action='store',
        required=True
    )

    parser.add_argument(
        '-k', '--key',
        help='OpenAI API key',
        action='store',
        required=True
    )

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f'[error] file {args.file} does not exist')
        sys.exit(1)

    if not os.path.isfile(args.file):
        print(f'[error] {args.file} is not a valid file')
        sys.exit(1)

    if not args.file.endswith('.pdf'):
        print(f'[error] {args.file} may not be a PDF file (check file extension)')
        sys.exit(1)

    chat(args.file, args.key)
