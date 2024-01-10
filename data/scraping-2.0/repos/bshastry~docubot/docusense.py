#!/usr/bin/env python3
"""
This script provides functionality for summarizing a given document using OpenAI's GPT-3.5-turbo model.
It includes the 'init()' function to initialize environment variables, the 'summarize()' function to generate summaries,
and the 'docusense()' function as the entry point for the script. The 'docusense()' function takes command-line arguments
for the document path, chunk size, and chunk overlap. It utilizes prompts and chains to perform the summarization process.
"""


def init():
    """
    Initializes the environment variables by loading the .env file.

    Returns:
    None
    """
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(), override=True)


def summarize(
    document: str,
    summary_file: str,
    chunk_size: int,
    chunk_overlap: int,
    max_single_shot_num_tokens: int = 2048,
) -> None:
    """
    Summarizes a given document using OpenAI's GPT-3.5-turbo model.

    Args:
        document (str): The path to the document to be summarized.
        chunk_size (int): The size of each chunk of the document to be summarized.
        chunk_overlap (int): The amount of overlap between each chunk of the document.
        max_single_shot_num_tokens (int, optional): The maximum number of tokens allowed for a single-shot summarization. Defaults to 2048.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified document path does not exist.
    """
    from langchain.chat_models import ChatOpenAI
    from langchain import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.chains.summarize import load_summarize_chain
    from document_loaders.document_loaders import (
        load_document,
        merge_document,
        chunk_data,
    )
    from text_utils.text_utils import num_tokens_and_cost

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    map_prompt = """
Write a concise summary of the following:
Text: `{text}`
CONCISE SUMMARY:
"""
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    combine_prompt = """
Write a concise summary of the following text that covers key points.
Add a title to the summary.
Start the summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
by BULLET POINTS if possible AND end the summary with a CONCLUSION.
Text: `{text}`
"""
    combine_prompt_template = PromptTemplate(
        template=combine_prompt, input_variables=["text"]
    )

    doc = load_document(document)
    num_tokens, cost = num_tokens_and_cost(doc)
    print(f"Approximate summarization cost: ${cost:.4f}")
    if num_tokens <= max_single_shot_num_tokens:
        chain = LLMChain(llm=llm, prompt=combine_prompt_template)
        print("Running single-shot summarization")
        summary = chain.run({"text": merge_document(doc)})
        print(f"Writing summary to {summary_file}... ", end="")
        with open(summary_file, "w") as f:
            f.write(summary)
        print("Done")
    else:
        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=combine_prompt_template,
        )
        print("Running multi-shot summarization")
        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=combine_prompt_template,
        )
        summary = chain.run(
            chunk_data(data=doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        print(f"Writing summary to {summary_file}... ", end="")
        with open(summary_file, "w") as f:
            f.write(summary)
        print("Done")


def docusense() -> None:
    """
    This function takes in a document path and summary file path and summarizes it using DocuSense.
    It also takes in optional arguments for chunk size and overlap.

    Returns:
        None
    """
    import argparse

    parser = argparse.ArgumentParser(description="DocuSense")
    parser.add_argument(
        "document", type=str, help="Path to the document to be summarized."
    )
    parser.add_argument(
        "summary_file",
        type=str,
        help="Path to the file where summary will be written to.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=3300, help="Chunk size in tokens."
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=100, help="Chunk overlap in tokens."
    )
    args = parser.parse_args()
    document = args.document
    summary_file = args.summary_file
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    print(f"Instantiating DocuSense for {document}")
    init()
    try:
        summarize(document, summary_file, chunk_size, chunk_overlap)
    except FileNotFoundError:
        print(f"File {document} not found")


if __name__ == "__main__":
    docusense()
