#!/usr/bin/env python
import logging
from gpt4all import GPT4All
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.prompts import PromptTemplate
import workaround


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    logging.debug("Loading model ...")
    # Try GPU first
    try:
        with workaround.suppress_stdout_stderr():
            model = GPT4All(
                "ggml-model-gpt4all-falcon-q4_0.bin",
                model_path="./models",
                allow_download=False,
                device="gpu",
            )
        logging.debug("Loaded for GPU.")
    except Exception as e:
        logging.debug(f"Loaded for GPU failed with: {e}.")
        with workaround.suppress_stdout_stderr():
            model = GPT4All(
                "ggml-model-gpt4all-falcon-q4_0.bin",
                model_path="./models",
                allow_download=False,
                device="cpu",
            )
        logging.debug("Loaded for CPU.")
    # Prepare query
    prompt_template = PromptTemplate.from_template(
        "Tell me about {content}. Do not exceed 42 tokens."
    )
    prompt = prompt_template.format(content="Hello World!")
    logging.debug("Start prompt ...")
    with workaround.suppress_stdout_stderr():
        output = model.generate(prompt, max_tokens=42, temp=0.1)
    logging.info(f"Answer 1 (with temp=0.1): {output}")
    with workaround.suppress_stdout_stderr():
        output = model.generate(prompt, max_tokens=42, temp=0.9)
    logging.info(f"Answer 2: (with temp=0.9) {output}")


if __name__ == "__main__":
    main()
